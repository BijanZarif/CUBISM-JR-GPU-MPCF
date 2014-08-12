/* *
 * GPUlab.cpp
 *
 * Created by Fabian Wermelinger on 5/28/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "GPUlab.h"
#include "Timer.h"
#include "MaxSpeedOfSound_CUDA.h"
#include "Convection_CUDA.h"
#include "Update_CUDA.h"

#include <string>
using std::string;

#ifdef _USE_HDF_
#include <hdf5.h>
#ifdef _FLOAT_PRECISION_
#define _HDF_REAL_ H5T_NATIVE_FLOAT
#else
#define _HDF_REAL_ H5T_NATIVE_DOUBLE
#endif
#endif


GPUlab::GPUlab(GridMPI& G, const uint_t nslices_, const int verbosity) :
    GPU_input_size( SLICE_GPU * (nslices_+6) ),
    GPU_output_size( SLICE_GPU * nslices_ ),
    nslices(nslices_), nslices_last( sizeZ % nslices_ ), nchunks( (sizeZ + nslices_ - 1) / nslices_ ),
    cart_world(G.getCartComm()), request(6), status(6),
    BUFFER1(GPU_input_size, GPU_output_size, 3*sizeY*nslices_, sizeX*3*nslices_, 0), // per chunk
    BUFFER2(GPU_input_size, GPU_output_size, 3*sizeY*nslices_, sizeX*3*nslices_, 0), // per chunk
    grid(G),
    halox(3*sizeY*sizeZ), // all domain (buffer zone for halo extraction + MPI send/recv)
    haloy(sizeX*3*sizeZ), // all domain
    haloz(sizeX*sizeY*3)  // all domain
{
    if (nslices_last != 0) // can be solved later
    {
        fprintf(stderr, "[GPUlab ERROR: CURRENTLY nslices MUST BE AN INTEGER MULTIPLE of GridMPI::sizeZ\n");
        exit(1);
    }

    if (0 < nslices_last && nslices_last < 3)
    {
        fprintf(stderr, "[GPUlab ERROR: Too few slices in the last chunk (have %d slices, should be 3 or higher)\n", nslices_last);
        exit(1);
    }

    chatty = QUIET;
    if (2 == verbosity) chatty = VERBOSE;

    _alloc_GPU();

    profiler = &GPU::profiler;

    _reset();

    curr_buffer = &BUFFER1; // are swapped with _swap_buffer()
    prev_buffer = &BUFFER2;

    prev_slices   = (1 - !nslices_last)*nslices_last + (!nslices_last)*nslices;
    prev_iz       = (nchunks-1) * nslices;
    prev_chunk_id = nchunks;

    int mycoords[3];
    grid.peindex(mycoords);
    for (int i = 0; i < 3; ++i)
    {
        myFeature[i*2 + 0] = mycoords[i] == 0 ? SKIN : FLESH;
        myFeature[i*2 + 1] = mycoords[i] == grid.getBlocksPerDimension(i)-1 ? SKIN : FLESH;
    }
    grid.getNeighborRanks(nbr);
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE
///////////////////////////////////////////////////////////////////////////////
template <index_map map>
void GPUlab::_copysend_halos(const int sender, Real * const cpybuf, const uint_t Nhalos, const int xS, const int xE, const int yS, const int yE, const int zS, const int zE)
{
    assert(Nhalos == (xE-xS)*(yE-yS)*(zE-zS));

#pragma omp parallel for
    for (int p = 0; p < GridMPI::NVAR; ++p)
    {
        const Real * const src = grid.pdata()[p];
        const uint_t offset = p * Nhalos;
        for (int iz = zS; iz < zE; ++iz)
            for (int iy = yS; iy < yE; ++iy)
                for (int ix = xS; ix < xE; ++ix)
                    cpybuf[offset + map(ix,iy,iz-zS)] = src[ix + sizeX * (iy + sizeY * iz)];
    }

    _issue_send(cpybuf, GridMPI::NVAR * Nhalos, sender);
}


void GPUlab::_copysend_halos(const int sender, Real * const cpybuf, const uint_t Nhalos, const int zS)
{
    assert(Nhalos == 3*SLICE_GPU);

    const uint_t srcoffset = SLICE_GPU * zS;
#pragma omp parallel for
    for (int p = 0; p < GridMPI::NVAR; ++p)
    {
        const Real * const src = grid.pdata()[p];
        const uint_t offset = p * Nhalos;
        memcpy(cpybuf + offset, src + srcoffset, Nhalos*sizeof(Real));
    }

    _issue_send(cpybuf, GridMPI::NVAR * Nhalos, sender);
}


void GPUlab::_alloc_GPU()
{
    GPU::alloc((void**) &maxSOS, nslices);
    gpu_allocation = ALLOCATED;
}


void GPUlab::_free_GPU()
{
    GPU::dealloc();
    gpu_allocation = FREE;
}


void GPUlab::_reset()
{
    if (nchunks == 1)
    {
        // whole chunk fits on the GPU
        chunk_state = SINGLE;
        curr_slices = sizeZ;
    }
    else
    {
        chunk_state = FIRST;
        curr_slices = nslices;
    }
    curr_iz = 0;
    curr_chunk_id = 1;
}


void GPUlab::_process_chunk_sos(const RealPtrVec_t& src)
{
    /* *
     * Processes a chunk for the maxSOS computation:
     * 1.) upload active buffer
     * 2.) launch GPU maxSOS kernel
     * 3.) ============== Initialize next chunk ===================
     * 4.) prepare data buffer for next chunk
     * 5.) wait until upload of active buffer has finished
     * */

    Timer timer;

    ///////////////////////////////////////////////////////////////////
    // 1.)
    ///////////////////////////////////////////////////////////////////
    /* GPU::h2d_3DArray(curr_buffer->GPUin, curr_slices); */
    GPU::h2d_3DArray(curr_buffer->GPUin, nslices, 0);

    ///////////////////////////////////////////////////////////////////
    // 2.)
    ///////////////////////////////////////////////////////////////////
    MaxSpeedOfSound_CUDA kernel;
    if (chatty) printf("\t[LAUNCH SOS KERNEL CHUNK %d]\n", curr_chunk_id);
    kernel.compute(curr_slices, 0);
    if (chatty) _end_info_current_chunk();

    ///////////////////////////////////////////////////////////////////
    // 3.)
    ///////////////////////////////////////////////////////////////////
    _init_next_chunk();
    if (chatty)
    {
        char title[256];
        sprintf(title, "MAX SOS PROCESSING CHUNK %d\n", curr_chunk_id);
        _start_info_current_chunk(title);
    }


    ///////////////////////////////////////////////////////////////////
    // 4.)
    ///////////////////////////////////////////////////////////////////
    const uint_t OFFSET = SLICE_GPU * curr_iz;
    timer.start();
    _copy_range(curr_buffer->GPUin, 0, src, OFFSET, SLICE_GPU * curr_slices);
    const double t1 = timer.stop();
    if (chatty) printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", curr_chunk_id, t1);

    ///////////////////////////////////////////////////////////////////
    // 5.)
    ///////////////////////////////////////////////////////////////////
    GPU::wait_event(GPU::H2D_3DARRAY);
    /* GPU::h2d_3DArray_wait(0); */
}


void GPUlab::_process_chunk_flow(const Real a, const Real b, const Real dtinvh, RealPtrVec_t& src, RealPtrVec_t& tmp)
{
    /* *
     * Process chunk for the RHS computation:
     * 1.)  Convert AOS->SOA tmp (hidden by h2d_3DArray)
     * 2.)  Upload GPU tmp (needed for divergence, uploaded on TMP stream)
     * 3.)  Launch GPU convection kernel
     * 4.)  Launch GPU update kernel
     * 5.)  Convert SOA->AOS of rhs and updated solution (applies to previous chunk, hidden by 2-4)
     * 6.)  ============== Initialize next chunk ===================
     * 7.)  Convert AOS->SOA for GPU input of new chunk (hidden by 2-4)
     * 8.)  Download GPU rhs of previous chunk into previous buffer (downloaded on TMP stream)
     * 9.)  Download GPU updated solution of previous chunk into previous buffer (downloaded on TMP stream)
     * 10.) Compute x/yghosts of new chunk and upload to GPU (overlaps download of 8-9, uploaded on MAIN stream)
     *      Upload GPU input of new chunk (3DArrays, overlaps download of 8-9, uploaded on MAIN stream)
     * */

    Timer timer;

    /* /////////////////////////////////////////////////////////////////// */
    /* // 1.) */
    /* /////////////////////////////////////////////////////////////////// */
    /* // TODO: NOT NEEDE ANYMORE */
    /* uint_t OFFSET = SLICE_GPU * curr_iz; */
    /* timer.start(); */
    /* _copy_range(curr_buffer->GPUtmp, 0, tmp, OFFSET, SLICE_GPU * curr_slices); */
    /* const double t1 = timer.stop(); */
    /* if (chatty) */
    /*     printf("\t[COPY TMP CHUNK %d TAKES %f sec]\n", curr_chunk_id, t1); */

    /* /////////////////////////////////////////////////////////////////// */
    /* // 2.) */
    /* /////////////////////////////////////////////////////////////////// */
    /* // TODO: NOT NEEDE ANYMORE */
    /* /1* GPU::h2d_tmp(curr_buffer->GPUtmp, SLICE_GPU * curr_slices); *1/ */
    /* GPU::h2d_tmp(curr_buffer->GPUtmp, GPU_output_size); */

    ///////////////////////////////////////////////////////////////////
    // 3.)
    ///////////////////////////////////////////////////////////////////
    /* _dump_chunk(1); */
    /* Convection_CUDA convection(a, dtinvh); */
    Convection_CUDA convection;

    if (chatty) printf("\t[LAUNCH CONVECTION KERNEL CHUNK %d]\n", curr_chunk_id);
    /* convection.compute(curr_slices, curr_iz); */
    //
    // overlap buf1 & buf2 ???
    /* profiler->push_start("FLUX KERNELS"); */
    convection.compute(curr_slices, 0, curr_buffer->stream_id);
    /* profiler->pop_stop(); */

    ///////////////////////////////////////////////////////////////////
    // 4.)
    ///////////////////////////////////////////////////////////////////
    /* // TODO: NOT NEEED */
    /* Update_CUDA update(b); */
    /* if (chatty) printf("\t[LAUNCH UPDATE KERNEL CHUNK %d]\n", curr_chunk_id); */
    /* update.compute(curr_slices); */

    ///////////////////////////////////////////////////////////////////
    // 5.)
    ///////////////////////////////////////////////////////////////////
    // TODO: MUST CHANGE -> DIVERGENCE + UPDATE
    switch (chunk_state)
    {
        // DL divF and update
        //
        // Since operations are on previous chunk, this must only be
        // done on INTERMEDIATE or LAST chunks.  The copy back of
        // SINGLE and (actual) LAST chunks must be done after
        // _process_chunk has finished processing all chunks.
        case INTERMEDIATE:
        case LAST: // operations are on chunk one before LAST (because of use of previous_buffer)!
            const uint_t prevOFFSET = SLICE_GPU * prev_iz;

            // wait for d2h_divF
            // (one kernel update(a, b, dtinvh, src, tmp, offset, divF, Nelements) )
            // compute tmp <- a * tmp - dtinvh * divF
            // update  src <- b * tmp + src
            //

            /* GPU::d2h_rhs_wait(); // make sure previous d2h has finished */
            /* timer.start(); */
            /* _copy_range(tmp, prevOFFSET, prev_buffer->GPUtmp, 0, SLICE_GPU * prev_slices); */
            /* const double t4 = timer.stop(); */
            /* if (chatty) */
            /*     printf("\t[COPY BACK TMP CHUNK %d TAKES %f sec]\n", prev_chunk_id, t4); */

            /* GPU::d2h_tmp_wait(); */
            /* timer.start(); */
            /* _copy_range(src, prevOFFSET, prev_buffer->GPUout, 0, SLICE_GPU * prev_slices); */
            /* const double t2 = timer.stop(); */
            /* if (chatty) */
            /*     printf("\t[COPY BACK OUTPUT CHUNK %d TAKES %f sec]\n", prev_chunk_id, t2); */

            break;
    }
    if (chatty) _end_info_current_chunk();

    ///////////////////////////////////////////////////////////////////
    // 6.)
    ///////////////////////////////////////////////////////////////////
    _init_next_chunk();
    if (chatty)
    {
        char title[256];
        sprintf(title, "RHS PROCESSING CHUNK %d\n", curr_chunk_id);
        _start_info_current_chunk(title);
    }

    ///////////////////////////////////////////////////////////////////
    // 7.)
    ///////////////////////////////////////////////////////////////////
    const uint_t OFFSET = SLICE_GPU * curr_iz;

    timer.start();
    switch (chunk_state)
    {
        /* // Prepare interior nodes for the next call to _process_all. */
        /* // The ghosts are copied later, once the new Lab has been */
        /* // loaded. */
        /* case FIRST: */
        /*     // interior + right ghosts */
        /*     printf("copy FIRST\n"); */
        /*     _copy_range(curr_buffer->GPUin, haloz.Nhalo, src, OFFSET, SLICE_GPU * curr_slices + haloz.Nhalo); */
        /*     break; */

        case INTERMEDIATE:
            {
                // left zghosts (reuse previous buffer)
                const uint_t prevOFFSET = SLICE_GPU * prev_slices;
                _copy_range(curr_buffer->GPUin, 0, prev_buffer->GPUin, prevOFFSET, haloz.Nhalo);

                // interior + right ghosts
                _copy_range(curr_buffer->GPUin, haloz.Nhalo, src, OFFSET, SLICE_GPU * curr_slices + haloz.Nhalo);
                break;
            }

        case LAST:
            {
                // left zghosts (reuse previous buffer)
                const uint_t prevOFFSET = SLICE_GPU * prev_slices;
                _copy_range(curr_buffer->GPUin, 0, prev_buffer->GPUin, prevOFFSET, haloz.Nhalo);

                // interior
                _copy_range(curr_buffer->GPUin, haloz.Nhalo, src, OFFSET, SLICE_GPU * curr_slices);

                // right zghosts
                const uint_t current_rightOFFSET = haloz.Nhalo + SLICE_GPU * curr_slices;
                _copy_range(curr_buffer->GPUin, current_rightOFFSET, haloz.right, 0, haloz.Nhalo);
                break;
            }
    }
    const double t3 = timer.stop();
    if (chatty)
        printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", curr_chunk_id, t3);

    ///////////////////////////////////////////////////////////////////
    // 8.)
    ///////////////////////////////////////////////////////////////////
    /* // TODO: REMOVE */
    /* /1* GPU::d2h_rhs(prev_buffer->GPUtmp, SLICE_GPU * prev_slices); *1/ */
    /* GPU::d2h_rhs(prev_buffer->GPUtmp, GPU_output_size); */

    ///////////////////////////////////////////////////////////////////
    // 9.)
    ///////////////////////////////////////////////////////////////////
    /* GPU::d2h_tmp(prev_buffer->GPUout, SLICE_GPU * prev_slices); */
    /* GPU::d2h_tmp(prev_buffer->GPUout, GPU_output_size); */
    GPU::d2h_divF(prev_buffer->GPUout, GPU_output_size, prev_buffer->stream_id);

    ///////////////////////////////////////////////////////////////////
    // 10.)
    ///////////////////////////////////////////////////////////////////
    switch (chunk_state)
    {
        case INTERMEDIATE:
        case LAST:
            assert(curr_buffer->Nxghost == 3 * sizeY * curr_slices);
            assert(curr_buffer->Nyghost == 3 * sizeX * curr_slices);
            _copy_xyghosts();
            GPU::upload_xy_ghosts(curr_buffer->Nxghost, curr_buffer->xghost_l, curr_buffer->xghost_r,
                    curr_buffer->Nyghost, curr_buffer->yghost_l, curr_buffer->yghost_r,
                    curr_buffer->stream_id);

            /* GPU::h2d_3DArray(curr_buffer->GPUin, curr_slices+6); */
            GPU::h2d_3DArray(curr_buffer->GPUin, nslices+6, curr_buffer->stream_id);

            break;
    }
}


void GPUlab::_init_next_chunk()
{
    prev_slices   = curr_slices;
    prev_iz       = curr_iz;
    prev_chunk_id = curr_chunk_id;

    curr_iz += curr_slices;
    ++curr_chunk_id;

    if (curr_chunk_id > nchunks)
        _reset();
    else if (curr_chunk_id == nchunks)
    {
        curr_slices = (1 - !nslices_last)*nslices_last + (!nslices_last)*nslices;
        chunk_state = LAST;
    }
    else
    {
        curr_slices = nslices;
        chunk_state = INTERMEDIATE;
    }

    // use a new host buffer
    _swap_buffer();

    // set the number of ghosts in x/y direction for this buffer
    curr_buffer->Nxghost = 3*sizeY*curr_slices;
    curr_buffer->Nyghost = sizeX*3*curr_slices;
}


void GPUlab::_dump_chunk(const int complete)
{
    static unsigned int ndumps = 0;
    printf("Dumping Chunk %d (total dumps %d)...\n", curr_chunk_id, ++ndumps);

    char fname[256];

    static const unsigned int NCHANNELS = 9;
    Real *array_all;
    hsize_t count[4];
    hsize_t dims[4];
    hsize_t offset[4] = {0, 0, 0, 0};

    if (complete)
    {
        sprintf(fname, "chunk-all_%02d-%04d", curr_chunk_id, ndumps);

        const unsigned int NX = GridMPI::sizeX+6;
        const unsigned int NY = GridMPI::sizeY+6;
        const unsigned int NZ = curr_slices+6;
        array_all = new Real[NX * NY * NZ * NCHANNELS];
        dims[0]=count[0]=NZ; dims[1]=count[1]=NY; dims[2]=count[2]=NX;
        dims[3]=count[3]=NCHANNELS;

        // set all NaN
        for (unsigned int i = 0; i < NX*NY*NZ*NCHANNELS; ++i)
            array_all[i] = NAN;

        // input buffer
        for(unsigned int iz = 0; iz < NZ; ++iz)
            for(unsigned int iy = 3; iy < NY-3; ++iy)
                for(unsigned int ix = 3; ix < NX-3; ++ix)
                {
                    Real output[NCHANNELS] = {0};

                    const unsigned int idx = ID3(ix-3,iy-3,iz,NX-6,NY-6);
                    for(unsigned int i = 0; i < curr_buffer->GPUin.size(); ++i)
                        output[i] = curr_buffer->GPUin[i][idx];

                    const unsigned int gidx = ID3(ix,iy,iz,NX,NY);
                    Real * const ptr = array_all + NCHANNELS*gidx;
                    for(unsigned int i=0; i<NCHANNELS; ++i)
                        ptr[i] = output[i];
                }
        // xghosts
        for(unsigned int iz = 3; iz < NZ-3; ++iz)
            for(unsigned int iy = 3; iy < NY-3; ++iy)
                for(unsigned int ix = 0; ix < 3; ++ix)
                {
                    Real output[NCHANNELS] = {0};
                    // left
                    for(unsigned int i = 0; i < curr_buffer->xghost_l.size(); ++i)
                        output[i] = curr_buffer->xghost_l[i][ghostmap::X(ix,iy-3,iz-3)];
                    Real * ptr = array_all + NCHANNELS*ID3(ix,iy,iz,NX,NY);
                    for(unsigned int i=0; i<NCHANNELS; ++i)
                        ptr[i] = output[i];
                    // right
                    for(unsigned int i = 0; i < curr_buffer->xghost_r.size(); ++i)
                        output[i] = curr_buffer->xghost_r[i][ghostmap::X(ix,iy-3,iz-3)];
                    ptr = array_all + NCHANNELS*ID3(ix+GridMPI::sizeX+3,iy,iz,NX,NY);
                    for(unsigned int i=0; i<NCHANNELS; ++i)
                        ptr[i] = output[i];
                }
        // yghosts
        for(unsigned int iz = 3; iz < NZ-3; ++iz)
            for(unsigned int iy = 0; iy < 3; ++iy)
                for(unsigned int ix = 3; ix < NX-3; ++ix)
                {
                    Real output[NCHANNELS] = {0};
                    // left
                    for(unsigned int i = 0; i < curr_buffer->yghost_l.size(); ++i)
                        output[i] = curr_buffer->yghost_l[i][ghostmap::Y(ix-3,iy,iz-3)];
                    Real * ptr = array_all + NCHANNELS*ID3(ix,iy,iz,NX,NY);
                    for(unsigned int i=0; i<NCHANNELS; ++i)
                        ptr[i] = output[i];
                    // right
                    for(unsigned int i = 0; i < curr_buffer->yghost_r.size(); ++i)
                        output[i] = curr_buffer->yghost_r[i][ghostmap::Y(ix-3,iy,iz-3)];
                    ptr = array_all + NCHANNELS*ID3(ix,iy+GridMPI::sizeY+3,iz,NX,NY);
                    for(unsigned int i=0; i<NCHANNELS; ++i)
                        ptr[i] = output[i];
                }
    }
    else
    {
        sprintf(fname, "chunk_%02d-%04d", curr_chunk_id, ndumps);

        const unsigned int NX = GridMPI::sizeX;
        const unsigned int NY = GridMPI::sizeY;
        const unsigned int NZ = curr_slices+6;
        array_all = new Real[NX * NY * NZ * NCHANNELS];
        dims[0]=count[0]=NZ; dims[1]=count[1]=NY; dims[2]=count[2]=NX;
        dims[3]=count[3]=NCHANNELS;

        for(unsigned int iz = 0; iz < NZ; ++iz)
            for(unsigned int iy = 0; iy < NY; ++iy)
                for(unsigned int ix = 0; ix < NX; ++ix)
                {
                    const unsigned int idx = ID3(ix,iy,iz,NX,NY);
                    Real output[NCHANNELS] = {0};

                    for(unsigned int i = 0; i < curr_buffer->GPUin.size(); ++i)
                        output[i] = curr_buffer->GPUin[i][idx];

                    Real * const ptr = array_all + NCHANNELS*idx;
                    for(unsigned int i=0; i<NCHANNELS; ++i)
                        ptr[i] = output[i];
                }
    }

    herr_t status;
    hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;

    const string basename(fname);

    H5open();
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    file_id = H5Fcreate((basename+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    status = H5Pclose(fapl_id);

    fapl_id = H5Pcreate(H5P_DATASET_XFER);
    fspace_id = H5Screate_simple(4, dims, NULL);
    dataset_id = H5Dcreate(file_id, "data", _HDF_REAL_, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    fspace_id = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    mspace_id = H5Screate_simple(4, count, NULL);
    status = H5Dwrite(dataset_id, _HDF_REAL_, mspace_id, fspace_id, fapl_id, array_all);

    status = H5Sclose(mspace_id);
    status = H5Sclose(fspace_id);
    status = H5Dclose(dataset_id);
    status = H5Pclose(fapl_id);
    status = H5Fclose(file_id);
    H5close();

    delete [] array_all;

    {
        FILE *xmf = 0;
        xmf = fopen((basename+".xmf").c_str(), "w");
        fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
        fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
        fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
        fprintf(xmf, " <Domain>\n");
        fprintf(xmf, "   <Grid GridType=\"Uniform\">\n");
        fprintf(xmf, "     <Time Value=\"%05d\"/>\n", curr_chunk_id);
        fprintf(xmf, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n", (int)dims[0], (int)dims[1], (int)dims[2]);
        fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
        fprintf(xmf, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
        if (complete)
            fprintf(xmf, "        %e %e %e\n", grid.getH()*((int)curr_iz-3), -3*grid.getH(), -3*grid.getH());
        else
            fprintf(xmf, "        %e %e %e\n", grid.getH()*((int)curr_iz-3), 0., 0.);
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
        fprintf(xmf, "        %e %e %e\n", grid.getH(), grid.getH(), grid.getH());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Geometry>\n");

        fprintf(xmf, "     <Attribute Name=\"data\" AttributeType=\"Tensor\" Center=\"Node\">\n");
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", (int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3]);
        fprintf(xmf, "        %s:/data\n", (basename+".h5").c_str());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");

        fprintf(xmf, "   </Grid>\n");
        fprintf(xmf, " </Domain>\n");
        fprintf(xmf, "</Xdmf>\n");
        fclose(xmf);
    }
}


void GPUlab::_print_array(const Real * const in, const uint_t size)
{
    for (uint_t i = 0; i < size; ++i)
        printf("%f ", in[i]);
    printf("\n");
}


void GPUlab::_show_feature()
{
    int c[3];
    grid.peindex(c);
    for (int i = 0; i < 3; ++i)
    {
        printf("MPI coords (%d,%d,%d) -- i = %d, left:  %s\n", c[0], c[1], c[2], i, myFeature[2*i+0] == SKIN ? "Skin" : "Flesh");
        printf("MPI coords (%d,%d,%d) -- i = %d, right: %s\n", c[0], c[1], c[2], i, myFeature[2*i+1] == SKIN ? "Skin" : "Flesh");
    }
}


void GPUlab::_start_info_current_chunk(const std::string title)
{
    std::string state;
    switch (chunk_state)
    {
        case FIRST:        state = "FIRST"; break;
        case INTERMEDIATE: state = "INTERMEDIATE"; break;
        case LAST:         state = "LAST"; break;
        case SINGLE:       state = "SINGLE"; break;
    }
    printf("{\n");
    printf("\t%s\n", title.c_str());
    printf("\t[CURRENT CHUNK:      \t%d/%d]\n", curr_chunk_id, nchunks);
    printf("\t[CURRENT CHUNK STATE:\t%s]\n", state.c_str());
    printf("\t[CURRENT SLICES:     \t%d/%d]\n", curr_slices, nslices);
    printf("\t[PREVIOUS SLICES:    \t%d/%d]\n", prev_slices, nslices);
    printf("\t[CURRENT Z-POS:      \t%d/%d]\n", curr_iz, sizeZ);
    printf("\t[NUMBER OF NODES:    \t%d]\n", SLICE_GPU * curr_slices);
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC
///////////////////////////////////////////////////////////////////////////////
void GPUlab::load_ghosts(const double t)
{
    // TODO: THIS NEEDS THOROUGH TESTING!

    if (myFeature[0] == FLESH) _copysend_halos<flesh2ghost::X_L>(0, &halox.send_left[0], halox.Nhalo, 0, 3, 0, sizeY, 0, sizeZ);
    if (myFeature[1] == FLESH) _copysend_halos<flesh2ghost::X_R>(1, &halox.send_right[0],halox.Nhalo, sizeX-3, sizeX, 0, sizeY, 0, sizeZ);
    if (myFeature[2] == FLESH) _copysend_halos<flesh2ghost::Y_L>(2, &haloy.send_left[0], haloy.Nhalo, 0, sizeX, 0, 3, 0, sizeZ);
    if (myFeature[3] == FLESH) _copysend_halos<flesh2ghost::Y_R>(3, &haloy.send_right[0],haloy.Nhalo, 0, sizeX, sizeY-3, sizeY, 0, sizeZ);
    if (myFeature[4] == FLESH) _copysend_halos(4, &haloz.send_left[0], haloz.Nhalo, 0);
    if (myFeature[5] == FLESH) _copysend_halos(5, &haloz.send_right[0],haloz.Nhalo, sizeZ-3);

    // need more digging here! TODO: use MPI_Irecv ? even better: hide MPI
    // latency with communicating per chunk on the host.  With this you can
    // hide MPI comm with GPU
    if (myFeature[0] == FLESH) _issue_recv(&halox.recv_left[0],  halox.Allhalos, 0); // x/yhalos directly into pinned mem and H2D
    if (myFeature[1] == FLESH) _issue_recv(&halox.recv_right[0], halox.Allhalos, 1);
    if (myFeature[2] == FLESH) _issue_recv(&haloy.recv_left[0],  haloy.Allhalos, 2);
    if (myFeature[3] == FLESH) _issue_recv(&haloy.recv_right[0], haloy.Allhalos, 3);
    if (myFeature[4] == FLESH) _issue_recv(&haloz.recv_left[0],  haloz.Allhalos, 4); // receive into curr_buffer->zghost_l ? why not
    if (myFeature[5] == FLESH) _issue_recv(&haloz.recv_right[0], haloz.Allhalos, 5);

    _apply_bc(t); // BC's apply to all myFeature == SKIN
}


double GPUlab::max_sos(float& sos)
{
    /* *
     * 1.) Init maxSOS = 0 (mapped integer -> zero-copy)
     * 2.) Copy data into input buffer for the FIRST/SINGLE chunk
     * 3.) Process all chunks
     * 4.) Synchronize stream to make sure reduction is complete
     * */

    RealPtrVec_t& src = grid.pdata();

    Timer tsos;
    tsos.start();

    ///////////////////////////////////////////////////////////////
    // 1.)
    ///////////////////////////////////////////////////////////////
    *maxSOS = 0;

    ///////////////////////////////////////////////////////////////
    // 2.)
    ///////////////////////////////////////////////////////////////
    Timer timer;
    timer.start();
    _copy_range(curr_buffer->GPUin, 0, src, 0, SLICE_GPU * curr_slices);
    const double t1 = timer.stop();
    if (chatty)
    {
        char title[256];
        sprintf(title, "MAX SOS PROCESSING CHUNK %d\n", curr_chunk_id);
        _start_info_current_chunk(title);
        printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", curr_chunk_id, t1);
    }

    ///////////////////////////////////////////////////////////////
    // 3.)
    ///////////////////////////////////////////////////////////////
    for (int i = 0; i < nchunks; ++i)
        _process_chunk_sos(src);
    if (chatty) _end_info_current_chunk();

    ///////////////////////////////////////////////////////////////
    // 4.)
    ///////////////////////////////////////////////////////////////
    this->_syncStream(GPU::streamID::S1);

    // maxSOS should be unsigned int, no?
    assert(sizeof(float) == sizeof(int));
    union {float f; int i;} ret;
    ret.i = *maxSOS;
    sos   = ret.f;

    return tsos.stop();
}


double GPUlab::process_all(const Real a, const Real b, const Real dtinvh)
{
    /* *
     * 1.) Extract x/yghosts for current chunk and upload to GPU
     * 2.) Copy ghosts and interior data into buffer for FIRST/SINGLE chunk
     * 3.) Upload GPU input for FIRST/SINGLE chunk (3DArrays)
     * 4.) Process all chunks
     * 5.) Copy back of GPU updated solution for LAST/SINGLE chunk
     * */

    RealPtrVec_t& src = grid.pdata();
    RealPtrVec_t& tmp = grid.ptmp();

    Timer tall;
    tall.start();

    if (chatty)
    {
        char title[256];
        sprintf(title, "RHS PROCESSING CHUNK %d\n", curr_chunk_id);
        _start_info_current_chunk(title);
    }

    ///////////////////////////////////////////////////////////////
    // 1.)
    ///////////////////////////////////////////////////////////////
    assert(curr_buffer->Nxghost == 3 * sizeY * curr_slices);
    assert(curr_buffer->Nyghost == 3 * sizeX * curr_slices);
    _copy_xyghosts();
    GPU::upload_xy_ghosts(curr_buffer->Nxghost, curr_buffer->xghost_l, curr_buffer->xghost_r,
            curr_buffer->Nyghost, curr_buffer->yghost_l, curr_buffer->yghost_r,
            curr_buffer->stream_id);

    ///////////////////////////////////////////////////////////////
    // 2.)
    ///////////////////////////////////////////////////////////////
    Timer timer;
    uint_t Nelements = SLICE_GPU * curr_slices;

    // copy left ghosts always (TODO: CAN BE DONE BY MPI RECV)
    _copy_range(curr_buffer->GPUin, 0, haloz.left, 0, haloz.Nhalo);
    switch (chunk_state) // right ghosts are conditional
    {
        case FIRST: Nelements += haloz.Nhalo; break;
        case SINGLE:
                    _copy_range(curr_buffer->GPUin, haloz.Nhalo + Nelements, haloz.right, 0, haloz.Nhalo);
                    break;
    }

    // interior data
    timer.start();
    _copy_range(curr_buffer->GPUin, haloz.Nhalo, src, 0, Nelements);
    const double t1 = timer.stop();
    if (chatty) printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", curr_chunk_id, t1);

    ///////////////////////////////////////////////////////////////
    // 3.)
    ///////////////////////////////////////////////////////////////
    /* GPU::h2d_3DArray(curr_buffer->GPUin, curr_slices+6); */
    GPU::h2d_3DArray(curr_buffer->GPUin, nslices+6, curr_buffer->stream_id);

    ///////////////////////////////////////////////////////////////
    // 4.)
    ///////////////////////////////////////////////////////////////
    for (int i = 0; i < nchunks; ++i)
        _process_chunk_flow(a, b, dtinvh, src, tmp);

    ///////////////////////////////////////////////////////////////
    // 5.)
    ///////////////////////////////////////////////////////////////
    const uint_t prevOFFSET = SLICE_GPU * prev_iz;

    // GPU rhs into tmp (d2h finishes first for rhs)
    GPU::d2h_rhs_wait();
    timer.start();
    _copy_range(tmp, prevOFFSET, prev_buffer->GPUtmp, 0, SLICE_GPU * prev_slices);
    const double t2 = timer.stop();
    if (chatty)
        printf("\t[COPY BACK TMP CHUNK %d TAKES %f sec]\n", prev_chunk_id, t2);

    // GPU update into src (a.k.a updated flow data)
    GPU::d2h_tmp_wait();
    timer.start();
    _copy_range(src, prevOFFSET, prev_buffer->GPUout, 0, SLICE_GPU * prev_slices);
    const double t3 = timer.stop();
    if (chatty)
        printf("\t[COPY BACK OUTPUT CHUNK %d TAKES %f sec]\n", prev_chunk_id, t3);

    if (chatty) _end_info_current_chunk();

    return tall.stop();
}

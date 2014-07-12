/* *
 * GPUlab.cpp
 *
 * Created by Fabian Wermelinger on 5/28/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "GPUlab.h"

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
    BUFFER1(GPU_input_size, GPU_output_size, 3*sizeY*nslices_, sizeX*3*nslices_), // per chunk
    BUFFER2(GPU_input_size, GPU_output_size, 3*sizeY*nslices_, sizeX*3*nslices_), // per chunk
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

    // farewell, brother
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

    // au revoir, soeur jumelle
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


void GPUlab::_dump_chunk()
{
    static unsigned int ndumps = 0;
    printf("Dumping Chunk %d (total dumps %d)...\n", curr_chunk_id, ++ndumps);

    char fname[256];
    sprintf(fname, "chunk_%02d-%04d.h5", curr_chunk_id, ndumps);

    herr_t status;
    hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;

    static const unsigned int NCHANNELS = 9;
    const unsigned int NX = GridMPI::sizeX;
    const unsigned int NY = GridMPI::sizeY;
    const unsigned int NZ = curr_slices+6;
    Real * array_all = new Real[NX * NY * NZ * NCHANNELS];

    hsize_t count[4]  = {NZ, NY, NX, NCHANNELS};
    hsize_t dims[4]   = {NZ, NY, NX, NCHANNELS};
    hsize_t offset[4] = {0, 0, 0, 0};

    H5open();
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    status = H5Pclose(fapl_id);

    for(unsigned int iz = 0; iz < NZ; ++iz)
        for(unsigned int iy = 0; iy < NY; ++iy)
            for(unsigned int ix = 0; ix < NX; ++ix)
            {
                const unsigned int idx = ID3(ix,iy,iz,NX,NY);
                Real output[NCHANNELS];
                for(unsigned int i=0; i<NCHANNELS; ++i) output[i] = 0;
                for(unsigned int i = 0; i < GridMPI::NVAR; ++i)
                    output[i] = curr_buffer->GPUin[i][idx];

                Real * const ptr = array_all + NCHANNELS*idx;
                for(unsigned int i=0; i<NCHANNELS; ++i)
                    ptr[i] = output[i];
            }

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
        char wrapper[256];
        sprintf(wrapper, "chunk_%02d-%04d.xmf", curr_chunk_id, ndumps);
        FILE *xmf = 0;
        xmf = fopen(wrapper, "w");
        fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
        fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
        fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
        fprintf(xmf, " <Domain>\n");
        fprintf(xmf, "   <Grid GridType=\"Uniform\">\n");
        fprintf(xmf, "     <Time Value=\"%05d\"/>\n", curr_chunk_id);
        fprintf(xmf, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n", (int)dims[0], (int)dims[1], (int)dims[2]);
        fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
        fprintf(xmf, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
        fprintf(xmf, "        %e %e %e\n", 0.,0.,grid.getH()*(curr_iz-3));
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
        fprintf(xmf, "        %e %e %e\n", grid.getH(), grid.getH(), grid.getH());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Geometry>\n");

        fprintf(xmf, "     <Attribute Name=\"data\" AttributeType=\"Tensor\" Center=\"Node\">\n");
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", (int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3]);
        fprintf(xmf, "        %s:/data\n",fname);
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
    if (myFeature[0] == FLESH) _issue_recv(&halox.recv_left[0],  halox.Allhalos, 0); // x/yhalos directly into pinned mem and H2D ALLLLLLL
    if (myFeature[1] == FLESH) _issue_recv(&halox.recv_right[0], halox.Allhalos, 1);
    if (myFeature[2] == FLESH) _issue_recv(&haloy.recv_left[0],  haloy.Allhalos, 2);
    if (myFeature[3] == FLESH) _issue_recv(&haloy.recv_right[0], haloy.Allhalos, 3);
    if (myFeature[4] == FLESH) _issue_recv(&haloz.recv_left[0],  haloz.Allhalos, 4); // receive into curr_buffer->zghost_l ? why not
    if (myFeature[5] == FLESH) _issue_recv(&haloz.recv_right[0], haloz.Allhalos, 5);

    _apply_bc(t); // BC's apply to all myFeature == SKIN
}

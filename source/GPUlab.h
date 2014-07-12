/* *
 * GPUlab.h
 *
 * Created by Fabian Wermelinger on 5/28/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include "GPU.h"
#include "GridMPI.h"
#include "Types.h"
#include "Timer.h"

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#ifdef _USE_HDF_
#include <hdf5.h>
#endif

#ifdef _FLOAT_PRECISION_
#define _MPI_REAL_ MPI_FLOAT
#else
#define _MPI_REAL_ MPI_DOUBLE
#endif


class GPUlab
{
    public:

        static const uint_t sizeX = GridMPI::sizeX;
        static const uint_t sizeY = GridMPI::sizeY;
        static const uint_t sizeZ = GridMPI::sizeZ;
        static const uint_t SLICE_GPU = GridMPI::sizeX * GridMPI::sizeY;


    private:

        enum {FIRST, INTERMEDIATE, LAST, SINGLE} chunk_state;
        enum {ALLOCATED, FREE} gpu_allocation;
        enum {QUIET=0, VERBOSE} chatty;

        // GPU BUFFER SIZES
        const uint_t GPU_input_size;
        const uint_t GPU_output_size;

        // CHUNK RELATED
        const uint_t nslices;
        const uint_t nslices_last; // nslices_last = sizeZ % nslices
        const uint_t nchunks;

        int* maxSOS; // pointer to mapped memory CPU/GPU

        ///////////////////////////////////////////////////////////////////////
        // HALOS / MPI COMMUNICATION
        ///////////////////////////////////////////////////////////////////////
        const MPI_Comm cart_world;
        std::vector<MPI_Request> request;
        std::vector<MPI_Status> status;

        int nbr[6]; // neighbor ranks

        struct Halo // hello halo
        {
            static const uint_t NVAR = GridMPI::NVAR; // number of variables in set
            const uint_t Nhalo;
            const uint_t Allhalos;
            std::vector<Real> send_left, send_right; // for position x1 < x2, then x1 = buf_left, x2 = buf_right
            std::vector<Real> recv_left, recv_right;
            RealPtrVec_t left, right;
            Halo(const uint_t sizeHalo) :
                Nhalo(sizeHalo), Allhalos(NVAR*sizeHalo),
                send_left(NVAR*sizeHalo, 0.0),
                send_right(NVAR*sizeHalo, 0.0),
                recv_left(NVAR*sizeHalo, 0.0),  left(NVAR, NULL),
                recv_right(NVAR*sizeHalo, 0.0), right(NVAR, NULL)
            {
                for (int i = 0; i < NVAR; ++i)
                {
                    // for convenience
                    left[i]  = &recv_left [i * Nhalo];
                    right[i] = &recv_right[i * Nhalo];
                }
            }
        };

        // MPI
        inline void _issue_send(const Real * const sendbuf, const uint_t Nelements, const uint_t sender)
        {
            // why is the send buffer not a const pointer?? 3.0 Standard says
            // different
            MPI_Isend(const_cast<Real * const>(sendbuf), Nelements, _MPI_REAL_, nbr[sender], sender, cart_world, &request[sender]);
        }

        inline void _issue_recv(Real * const recvbuf, const uint_t Nelements, const uint_t receiver)
        {
            MPI_Recv(recvbuf, Nelements, _MPI_REAL_, nbr[receiver], MPI_ANY_TAG, cart_world, &status[receiver]);
        }

        // Halo extraction
        template <index_map map>
        void _copysend_halos(const int sender, Real * const cpybuf, const uint_t Nhalos, const int xS, const int xE, const int yS, const int yE, const int zS, const int zE);
        void _copysend_halos(const int sender, Real * const cpybuf, const uint_t Nhalos, const int zS);


        ///////////////////////////////////////////////////////////////////////
        // HOST BUFFERS (pinned)
        ///////////////////////////////////////////////////////////////////////
        struct HostBuffer
        {
            static const uint_t NVAR = GridMPI::NVAR; // number of variables in set
            const uint_t _sizeIn, _sizeOut;
            uint_t Nxghost, Nyghost; // may change depending on last chunk

            // Tmp storage for GPU input data
            cuda_vector_t GPUin_all;
            RealPtrVec_t GPUin;

            // Tmp storage for GPU tmp
            cuda_vector_t GPUtmp_all;
            RealPtrVec_t GPUtmp;

            // Tmp storage for GPU output data (updated solution)
            cuda_vector_t GPUout_all;
            RealPtrVec_t GPUout;

            // compact ghosts
            cuda_vector_t xyghost_all;
            RealPtrVec_t xghost_l, xghost_r, yghost_l, yghost_r;

            HostBuffer(const uint_t sizeIn, const uint_t sizeOut, const uint_t sizeXghost, const uint_t sizeYghost) :
                _sizeIn(sizeIn), _sizeOut(sizeOut),
                Nxghost(sizeXghost), Nyghost(sizeYghost),
                GPUin_all(NVAR*sizeIn, 0.0), GPUin(NVAR, NULL),
                GPUtmp_all(NVAR*sizeOut, 0.0), GPUtmp(NVAR, NULL),
                GPUout_all(NVAR*sizeOut, 0.0), GPUout(NVAR, NULL),
                xyghost_all(2*NVAR*sizeXghost + 2*NVAR*sizeYghost, 0.0),
                xghost_l(NVAR, NULL), xghost_r(NVAR, NULL),
                yghost_l(NVAR, NULL), yghost_r(NVAR, NULL)
            {
                for (uint_t i = 0; i < NVAR; ++i)
                {
                    GPUin[i]  = &GPUin_all[i * sizeIn];
                    GPUtmp[i] = &GPUtmp_all[i * sizeOut];
                    GPUout[i] = &GPUout_all[i * sizeOut];
                }
                realign_ghost_pointer(sizeXghost, sizeYghost);
            }

            // --> CURRENTLY NOT USED ELSEWHERE, THEREFORE REQUIRES sizeZ % nslices == 0
            // this is dangerous!  Realign the ghost buffer pointers which
            // allows to copy a reduced buffer size to the GPU, e.g. if nslices_last !=
            // 0 for the last chunk in the queue. (Data in the buffers will be
            // coalesced at all times.  Ghosts are stored all in the same
            // contiguous array for better H2D/D2H bandwidth)
            void realign_ghost_pointer(const uint_t sizeXg, const uint_t sizeYg)
            {
                assert(sizeXg  <= Nxghost);
                assert(sizeYg  <= Nyghost);

                const uint_t allXghost = 2*NVAR*sizeXg;
                for (uint_t i = 0; i < NVAR; ++i)
                {
                    xghost_l[i] = &xyghost_all[(0*NVAR + i) * sizeXg];
                    xghost_r[i] = &xyghost_all[(1*NVAR + i) * sizeXg];
                    yghost_l[i] = &xyghost_all[(0*NVAR + i) * sizeYg + allXghost];
                    yghost_r[i] = &xyghost_all[(1*NVAR + i) * sizeYg + allXghost];
                }
            }
        };

        // using 2 buffers for additional CPU overlap
        HostBuffer BUFFER1, BUFFER2;
        HostBuffer *curr_buffer; // active buffer
        HostBuffer *prev_buffer;
        inline void _swap_buffer() // switch active buffer
        {
            curr_buffer = ((prev_buffer = curr_buffer) == &BUFFER1 ? &BUFFER2 : &BUFFER1);
        }


        ///////////////////////////////////////////////////////////////////////
        // PRIVATE HELPER
        ///////////////////////////////////////////////////////////////////////
        void _alloc_GPU();
        void _free_GPU();
        inline void _syncGPU() { GPU::syncGPU(); }
        inline void _syncStream(GPU::streamID s) { GPU::syncStream(s); }
        void _reset();
        void _init_next_chunk();
        void _dump_chunk(const int complete = 0);

        inline void _copy_range(RealPtrVec_t& dst, const uint_t dstOFFSET, const RealPtrVec_t& src, const uint_t srcOFFSET, const uint_t Nelements)
        {
            for (int i = 0; i < GridMPI::NVAR; ++i)
                memcpy(dst[i] + dstOFFSET, src[i] + srcOFFSET, Nelements*sizeof(Real));
        }

        inline void _copy_xyghosts() // alternatively, copy ALL x/yghosts at beginning
        {
            // copy from the halos into the ghost buffer of the current chunk
            _copy_range(curr_buffer->xghost_l, 0, halox.left,  3*sizeY*curr_iz, curr_buffer->Nxghost);
            _copy_range(curr_buffer->xghost_r, 0, halox.right, 3*sizeY*curr_iz, curr_buffer->Nxghost);
            _copy_range(curr_buffer->yghost_l, 0, haloy.left,  3*sizeX*curr_iz, curr_buffer->Nyghost);
            _copy_range(curr_buffer->yghost_r, 0, haloy.right, 3*sizeX*curr_iz, curr_buffer->Nyghost);
        }


        // execution helper
        template <typename Ksos>
        void _process_chunk(const RealPtrVec_t& src)
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
            GPU::h2d_3DArray(curr_buffer->GPUin, nslices);

            ///////////////////////////////////////////////////////////////////
            // 2.)
            ///////////////////////////////////////////////////////////////////
            Ksos kernel;
            if (chatty) printf("\t[LAUNCH SOS KERNEL CHUNK %d]\n", curr_chunk_id);
            kernel.compute(curr_slices);
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
            GPU::h2d_3DArray_wait();
        }


        template <typename Kflow, typename Kupdate>
        void _process_chunk(const Real a, const Real b, const Real dtinvh, RealPtrVec_t& src, RealPtrVec_t& tmp)
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

            ///////////////////////////////////////////////////////////////////
            // 1.)
            ///////////////////////////////////////////////////////////////////
            uint_t OFFSET = SLICE_GPU * curr_iz;
            timer.start();
            _copy_range(curr_buffer->GPUtmp, 0, tmp, OFFSET, SLICE_GPU * curr_slices);
            const double t1 = timer.stop();
            if (chatty)
                printf("\t[COPY TMP CHUNK %d TAKES %f sec]\n", curr_chunk_id, t1);

            ///////////////////////////////////////////////////////////////////
            // 2.)
            ///////////////////////////////////////////////////////////////////
            /* GPU::h2d_tmp(curr_buffer->GPUtmp, SLICE_GPU * curr_slices); */
            GPU::h2d_tmp(curr_buffer->GPUtmp, GPU_output_size);

            ///////////////////////////////////////////////////////////////////
            // 3.)
            ///////////////////////////////////////////////////////////////////
             _dump_chunk(1);
            Kflow convection(a, dtinvh);
            if (chatty) printf("\t[LAUNCH CONVECTION KERNEL CHUNK %d]\n", curr_chunk_id);
            /* convection.compute(curr_slices, curr_iz); */
            convection.compute(curr_slices, 0);

            ///////////////////////////////////////////////////////////////////
            // 4.)
            ///////////////////////////////////////////////////////////////////
            Kupdate update(b);
            if (chatty) printf("\t[LAUNCH UPDATE KERNEL CHUNK %d]\n", curr_chunk_id);
            update.compute(curr_slices);

            ///////////////////////////////////////////////////////////////////
            // 5.)
            ///////////////////////////////////////////////////////////////////
            switch (chunk_state)
            {
                // Since operations are on previous chunk, this must only be
                // done on INTERMEDIATE or LAST chunks.  The copy back of
                // SINGLE and (actual) LAST chunks must be done after
                // _process_chunk has finished processing all chunks.
                case INTERMEDIATE:
                case LAST: // operations are on chunk one before LAST (because of use of previous_buffer)!
                    const uint_t prevOFFSET = SLICE_GPU * prev_iz;

                    GPU::d2h_rhs_wait(); // make sure previous d2h has finished
                    timer.start();
                    _copy_range(tmp, prevOFFSET, prev_buffer->GPUtmp, 0, SLICE_GPU * prev_slices);
                    const double t4 = timer.stop();
                    if (chatty)
                        printf("\t[COPY BACK TMP CHUNK %d TAKES %f sec]\n", prev_chunk_id, t4);

                    GPU::d2h_tmp_wait();
                    timer.start();
                    _copy_range(src, prevOFFSET, prev_buffer->GPUout, 0, SLICE_GPU * prev_slices);
                    const double t2 = timer.stop();
                    if (chatty)
                        printf("\t[COPY BACK OUTPUT CHUNK %d TAKES %f sec]\n", prev_chunk_id, t2);

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
            OFFSET = SLICE_GPU * curr_iz;

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
                        // left ghosts (reuse previous buffer)
                        const uint_t prevOFFSET = SLICE_GPU * prev_slices;
                        _copy_range(curr_buffer->GPUin, 0, prev_buffer->GPUin, prevOFFSET, haloz.Nhalo);

                        // interior + right ghosts
                        _copy_range(curr_buffer->GPUin, haloz.Nhalo, src, OFFSET, SLICE_GPU * curr_slices + haloz.Nhalo);
                        break;
                    }

                case LAST:
                    {
                        // left ghosts (reuse previous buffer)
                        const uint_t prevOFFSET = SLICE_GPU * prev_slices;
                        _copy_range(curr_buffer->GPUin, 0, prev_buffer->GPUin, prevOFFSET, haloz.Nhalo);

                        // interior
                        _copy_range(curr_buffer->GPUin, haloz.Nhalo, src, OFFSET, SLICE_GPU * curr_slices);

                        // right ghosts
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
            /* GPU::d2h_rhs(prev_buffer->GPUtmp, SLICE_GPU * prev_slices); */
            GPU::d2h_rhs(prev_buffer->GPUtmp, GPU_output_size);

            ///////////////////////////////////////////////////////////////////
            // 9.)
            ///////////////////////////////////////////////////////////////////
            /* GPU::d2h_tmp(prev_buffer->GPUout, SLICE_GPU * prev_slices); */
            GPU::d2h_tmp(prev_buffer->GPUout, GPU_output_size);

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
                                          curr_buffer->Nyghost, curr_buffer->yghost_l, curr_buffer->yghost_r);

                    /* GPU::h2d_3DArray(curr_buffer->GPUin, curr_slices+6); */
                    GPU::h2d_3DArray(curr_buffer->GPUin, nslices+6);

                    break;
            }
        }

        // info
        void _print_array(const Real * const in, const uint_t size);
        void _show_feature();
        void _start_info_current_chunk(const std::string title = "");
        inline void _end_info_current_chunk()
        {
            printf("}\n");
        }


    protected:

        enum {SKIN, FLESH} myFeature[6];
        GridMPI& grid;

        // CHUNK metrics
        uint_t curr_slices;
        uint_t curr_iz;
        uint_t curr_chunk_id;
        uint_t prev_slices;
        uint_t prev_iz;
        uint_t prev_chunk_id;

        // ghosts
        Halo halox, haloy, haloz;

        // boundary conditions (applied for myFeature == SKIN)
        virtual void _apply_bc(const double t = 0) {}


    public:

        GPUlab(GridMPI& G, const uint_t nslices, const int verbosity=0);
        virtual ~GPUlab() { _free_GPU(); }

        ///////////////////////////////////////////////////////////////////////
        // PUBLIC ACCESSORS
        ///////////////////////////////////////////////////////////////////////
        void load_ghosts(const double t = 0);


        template <typename Ksos>
        double max_sos(float& sos)
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
                _process_chunk<Ksos>(src);
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


        template <typename Kflow, typename Kupdate>
        double process_all(const Real a, const Real b, const Real dtinvh)
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
                                  curr_buffer->Nyghost, curr_buffer->yghost_l, curr_buffer->yghost_r);

            ///////////////////////////////////////////////////////////////
            // 2.)
            ///////////////////////////////////////////////////////////////
            Timer timer;
            uint_t Nelements = SLICE_GPU * curr_slices;

            // copy left ghosts always (CAN BE DONE BY MPI RECV)
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
            GPU::h2d_3DArray(curr_buffer->GPUin, nslices+6);

            ///////////////////////////////////////////////////////////////
            // 4.)
            ///////////////////////////////////////////////////////////////
            for (int i = 0; i < nchunks; ++i)
                _process_chunk<Kflow, Kupdate>(a, b, dtinvh, src, tmp);

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


        // info
        inline uint_t number_of_chunks() const { return nchunks; }
        inline uint_t chunk_slices() const { return curr_slices; }
        inline uint_t chunk_start_iz() const { return curr_iz; }
        inline uint_t chunk_id() const { return curr_chunk_id; }
};

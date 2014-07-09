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

#ifndef _PAGEABLE_HOST_MEM_
#include "cudaHostAllocator.h"
typedef std::vector<Real, cudaHostAllocator<Real> > cuda_vector_t;
#else
typedef std::vector<Real> cuda_vector_t;
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
        const uint_t CHUNK_LENGTH;
        const uint_t REM;
        const uint_t N_chunks;

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
            cuda_vector_t SOAin_all;
            RealPtrVec_t SOAin;

            // Tmp storage for GPU tmp
            cuda_vector_t SOAtmp_all;
            RealPtrVec_t SOAtmp;

            // Tmp storage for GPU output data (updated solution)
            cuda_vector_t SOAout_all;
            RealPtrVec_t SOAout;

            // compact ghosts
            cuda_vector_t xyghost_all;
            RealPtrVec_t xghost_l, xghost_r, yghost_l, yghost_r;

            HostBuffer(const uint_t sizeIn, const uint_t sizeOut, const uint_t sizeXghost, const uint_t sizeYghost) :
                _sizeIn(sizeIn), _sizeOut(sizeOut),
                Nxghost(sizeXghost), Nyghost(sizeYghost),
                SOAin_all(NVAR*sizeIn, 0.0), SOAin(NVAR, NULL),
                SOAtmp_all(NVAR*sizeOut, 0.0), SOAtmp(NVAR, NULL),
                SOAout_all(NVAR*sizeOut, 0.0), SOAout(NVAR, NULL),
                xyghost_all(2*NVAR*sizeXghost + 2*NVAR*sizeYghost, 0.0),
                xghost_l(NVAR, NULL), xghost_r(NVAR, NULL),
                yghost_l(NVAR, NULL), yghost_r(NVAR, NULL)
            {
                for (uint_t i = 0; i < NVAR; ++i)
                {
                    SOAin[i]  = &SOAin_all[i * sizeIn];
                    SOAtmp[i] = &SOAtmp_all[i * sizeOut];
                    SOAout[i] = &SOAout_all[i * sizeOut];
                }
                realign_ghost_pointer(sizeXghost, sizeYghost);
            }

            // --> CURRENTLY NOT USED ELSEWHERE, THEREFORE REQUIRES sizeZ % CHUNK_LENGTH == 0
            // this is dangerous!  Realign the ghost buffer pointers which
            // allows to copy a reduced buffer size to the GPU, e.g. if REM !=
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

        // using 2 SOA buffers for additional CPU overlap
        HostBuffer BUFFER1, BUFFER2;
        HostBuffer *buffer; // active buffer
        HostBuffer *previous_buffer;
        inline void _swap_buffer() // switch active buffer
        {
            buffer = ((previous_buffer = buffer) == &BUFFER1 ? &BUFFER2 : &BUFFER1);
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

        inline void _copy_range(RealPtrVec_t& dst, const uint_t dstOFFSET, const RealPtrVec_t& src, const uint_t srcOFFSET, const uint_t Nelements)
        {
            for (int i = 0; i < GridMPI::NVAR; ++i)
                memcpy(dst[i] + dstOFFSET, src[i] + srcOFFSET, Nelements*sizeof(Real));
        }

        inline void _copy_xyghosts() // alternatively, copy ALL x/yghosts at beginning
        {
            // copy from the halos into the ghost buffer of the current chunk
            _copy_range(buffer->xghost_l, 0, halox.left,  3*sizeY*current_iz, buffer->Nxghost);
            _copy_range(buffer->xghost_r, 0, halox.right, 3*sizeY*current_iz, buffer->Nxghost);
            _copy_range(buffer->yghost_l, 0, haloy.left,  3*sizeX*current_iz, buffer->Nyghost);
            _copy_range(buffer->yghost_r, 0, haloy.right, 3*sizeX*current_iz, buffer->Nyghost);
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
             * 4.) convert AOS->SOA data for new chunk
             * 5.) wait until upload of active buffer has finished
             * */

            Timer timer;

            ///////////////////////////////////////////////////////////////////
            // 1.)
            ///////////////////////////////////////////////////////////////////
            /* GPU::h2d_3DArray(buffer->SOA, sizeX, sizeY, current_length); */
            GPU::h2d_3DArray(buffer->SOAin, sizeX, sizeY, CHUNK_LENGTH);

            ///////////////////////////////////////////////////////////////////
            // 2.)
            ///////////////////////////////////////////////////////////////////
            Ksos kernel;
            if (chatty) printf("\t[LAUNCH SOS KERNEL CHUNK %d]\n", current_chunk_id);
            kernel.compute(sizeX, sizeY, current_length);
            if (chatty) _end_info_current_chunk();

            ///////////////////////////////////////////////////////////////////
            // 3.)
            ///////////////////////////////////////////////////////////////////
            _init_next_chunk();
            if (chatty)
            {
                char title[256];
                sprintf(title, "MAX SOS PROCESSING CHUNK %d\n", current_chunk_id);
                _start_info_current_chunk(title);
            }


            ///////////////////////////////////////////////////////////////////
            // 4.)
            ///////////////////////////////////////////////////////////////////
            const uint_t OFFSET = SLICE_GPU * current_iz;
            timer.start();

            _copy_range(buffer->SOAin, 0, src, OFFSET, SLICE_GPU * current_length);

            /* switch (chunk_state) */
            /* { */
            /*     /1* case FIRST: // Pre-copy for rhs coputation after maxSOS *1/ */
            /*     /1*     // interior + right ghosts *1/ */
            /*     /1*     _copy_range(buffer->SOAin, haloz.Nhalo, src, OFFSET, SLICE_GPU * current_length + haloz.Nhalo); *1/ */
            /*     /1*     break; *1/ */

            /*     /1* case SINGLE: // Pre-copy for rhs computation after maxSOS *1/ */
            /*     /1*     // interior only *1/ */
            /*     /1*     _copy_range(buffer->SOAin, haloz.Nhalo, src, OFFSET, SLICE_GPU * current_length); *1/ */
            /*     /1*     break; *1/ */

            /*     /1* default: // copy next chunk for maxSOS *1/ */
            /*     /1*     _copy_range(buffer->SOAin, 0, src, OFFSET, SLICE_GPU * current_length); *1/ */
            /*     /1*     break; *1/ */
            /* } */
            const double t1 = timer.stop();
            if (chatty) printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", current_chunk_id, t1);

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
            uint_t OFFSET = SLICE_GPU * current_iz;
            timer.start();
            _copy_range(buffer->SOAtmp, 0, tmp, OFFSET, SLICE_GPU * current_length);
            const double t1 = timer.stop();
            if (chatty)
                printf("\t[COPY TMP CHUNK %d TAKES %f sec]\n", current_chunk_id, t1);

            ///////////////////////////////////////////////////////////////////
            // 2.)
            ///////////////////////////////////////////////////////////////////
            /* GPU::h2d_tmp(buffer->SOAtmp, SLICE_GPU * current_length); */
            GPU::h2d_tmp(buffer->SOAtmp, GPU_output_size);

            ///////////////////////////////////////////////////////////////////
            // 3.)
            ///////////////////////////////////////////////////////////////////
            Kflow convection(a, dtinvh);
            if (chatty) printf("\t[LAUNCH CONVECTION KERNEL CHUNK %d]\n", current_chunk_id);
            /* convection.compute(sizeX, sizeY, current_length, current_iz); */
            convection.compute(sizeX, sizeY, current_length, 0);

            ///////////////////////////////////////////////////////////////////
            // 4.)
            ///////////////////////////////////////////////////////////////////
            Kupdate update(b);
            if (chatty) printf("\t[LAUNCH UPDATE KERNEL CHUNK %d]\n", current_chunk_id);
            update.compute(sizeX, sizeY, current_length);

            ///////////////////////////////////////////////////////////////////
            // 5.)
            ///////////////////////////////////////////////////////////////////
            switch (chunk_state)
            {
                // Since operations are on previous chunk, this must only be
                // done on INTERMEDIATE or LAST chunks.  The SOA->AOS copy back
                // of SINGLE and (actual) LAST chunks must be done after
                // _process_chunk has finished processing all chunks.
                case INTERMEDIATE:
                case LAST: // operations are on chunk one before LAST (because of use of previous_buffer)!
                    const uint_t prevOFFSET = SLICE_GPU * previous_iz;

                    GPU::d2h_rhs_wait(); // make sure previous d2h has finished
                    timer.start();
                    _copy_range(tmp, prevOFFSET, previous_buffer->SOAtmp, 0, SLICE_GPU * previous_length);
                    const double t4 = timer.stop();
                    if (chatty)
                        printf("\t[COPY BACK TMP CHUNK %d TAKES %f sec]\n", previous_chunk_id, t4);

                    GPU::d2h_tmp_wait();
                    timer.start();
                    _copy_range(src, prevOFFSET, previous_buffer->SOAout, 0, SLICE_GPU * previous_length);
                    const double t2 = timer.stop();
                    if (chatty)
                        printf("\t[COPY BACK OUTPUT CHUNK %d TAKES %f sec]\n", previous_chunk_id, t2);

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
                sprintf(title, "RHS PROCESSING CHUNK %d\n", current_chunk_id);
                _start_info_current_chunk(title);
            }

            ///////////////////////////////////////////////////////////////////
            // 7.)
            ///////////////////////////////////////////////////////////////////
            OFFSET = SLICE_GPU * current_iz;

            timer.start();
            switch (chunk_state)
            {
                /* // Prepare interior nodes for the next call to _process_all. */
                /* // The ghosts are copied later, once the new Lab has been */
                /* // loaded. */
                /* case FIRST: */
                /*     // interior + right ghosts */
                /*     printf("copy FIRST\n"); */
                /*     _copy_range(buffer->SOAin, haloz.Nhalo, src, OFFSET, SLICE_GPU * current_length + haloz.Nhalo); */
                /*     break; */

                case INTERMEDIATE:
                    {
                        // left ghosts (reuse previous buffer)
                        const uint_t prevOFFSET = SLICE_GPU * previous_length;
                        _copy_range(buffer->SOAin, 0, previous_buffer->SOAin, prevOFFSET, haloz.Nhalo);

                        // interior + right ghosts
                        _copy_range(buffer->SOAin, haloz.Nhalo, src, OFFSET, SLICE_GPU * current_length + haloz.Nhalo);
                        break;
                    }

                case LAST:
                    {
                        // left ghosts (reuse previous buffer)
                        const uint_t prevOFFSET = SLICE_GPU * previous_length;
                        _copy_range(buffer->SOAin, 0, previous_buffer->SOAin, prevOFFSET, haloz.Nhalo);

                        // interior
                        _copy_range(buffer->SOAin, haloz.Nhalo, src, OFFSET, SLICE_GPU * current_length);

                        // right ghosts
                        const uint_t current_rightOFFSET = haloz.Nhalo + SLICE_GPU * current_length;
                        _copy_range(buffer->SOAin, current_rightOFFSET, haloz.right, 0, haloz.Nhalo);
                        break;
                    }
            }
            const double t3 = timer.stop();
            if (chatty)
                printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", current_chunk_id, t3);

            ///////////////////////////////////////////////////////////////////
            // 8.)
            ///////////////////////////////////////////////////////////////////
            /* GPU::d2h_rhs(previous_buffer->SOAtmp, SLICE_GPU * previous_length); */
            GPU::d2h_rhs(previous_buffer->SOAtmp, GPU_output_size);

            ///////////////////////////////////////////////////////////////////
            // 9.)
            ///////////////////////////////////////////////////////////////////
            /* GPU::d2h_tmp(previous_buffer->SOAout, SLICE_GPU * previous_length); */
            GPU::d2h_tmp(previous_buffer->SOAout, GPU_output_size);

            ///////////////////////////////////////////////////////////////////
            // 10.)
            ///////////////////////////////////////////////////////////////////
            switch (chunk_state)
            {
                case INTERMEDIATE:
                case LAST:
                    assert(buffer->Nxghost == 3 * sizeY * current_length);
                    assert(buffer->Nyghost == 3 * sizeX * current_length);
                    _copy_xyghosts();
                    GPU::upload_xy_ghosts(buffer->Nxghost, buffer->xghost_l, buffer->xghost_r,
                                          buffer->Nyghost, buffer->yghost_l, buffer->yghost_r);

                    /* GPU::h2d_3DArray(buffer->SOAin, sizeX, sizeY, current_length+6); */
                    GPU::h2d_3DArray(buffer->SOAin, sizeX, sizeY, CHUNK_LENGTH+6);

                    break;
            }
        }

        // info
        void _printSOA(const Real * const in, const uint_t size);
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
        uint_t current_length;
        uint_t current_iz;
        uint_t current_chunk_id;
        uint_t previous_length;
        uint_t previous_iz;
        uint_t previous_chunk_id;

        // ghosts
        Halo halox, haloy, haloz;

        // boundary conditions (applied for myFeature == SKIN)
        virtual void _apply_bc(const double t = 0) {}


    public:

        GPUlab(GridMPI& G, const uint_t CL);
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
             * 2.) Convert AOS->SOA for the FIRST/SINGLE chunk
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
            _copy_range(buffer->SOAin, 0, src, 0, SLICE_GPU * current_length);
            const double t1 = timer.stop();
            if (chatty)
            {
                char title[256];
                sprintf(title, "MAX SOS PROCESSING CHUNK %d\n", current_chunk_id);
                _start_info_current_chunk(title);
                printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", current_chunk_id, t1);
            }

            ///////////////////////////////////////////////////////////////
            // 3.)
            ///////////////////////////////////////////////////////////////
            for (int i = 0; i < N_chunks; ++i)
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
                sprintf(title, "RHS PROCESSING CHUNK %d\n", current_chunk_id);
                _start_info_current_chunk(title);
            }

            ///////////////////////////////////////////////////////////////
            // 1.)
            ///////////////////////////////////////////////////////////////
            assert(buffer->Nxghost == 3 * sizeY * current_length);
            assert(buffer->Nyghost == 3 * sizeX * current_length);
            _copy_xyghosts();
            GPU::upload_xy_ghosts(buffer->Nxghost, buffer->xghost_l, buffer->xghost_r,
                                  buffer->Nyghost, buffer->yghost_l, buffer->yghost_r);

            ///////////////////////////////////////////////////////////////
            // 2.)
            ///////////////////////////////////////////////////////////////
            Timer timer;
            uint_t Nelements = SLICE_GPU * current_length;

            // copy left ghosts always (CAN BE DONE BY MPI RECV)
            _copy_range(buffer->SOAin, 0, haloz.left, 0, haloz.Nhalo);
            switch (chunk_state) // right ghosts are conditional
            {
                case FIRST: Nelements += haloz.Nhalo; break;
                case SINGLE:
                    _copy_range(buffer->SOAin, haloz.Nhalo + Nelements, haloz.right, 0, haloz.Nhalo);
                    break;
            }

            // interior data
            timer.start();
            _copy_range(buffer->SOAin, haloz.Nhalo, src, 0, Nelements);
            const double t1 = timer.stop();
            if (chatty) printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", current_chunk_id, t1);

            ///////////////////////////////////////////////////////////////
            // 3.)
            ///////////////////////////////////////////////////////////////
            /* GPU::h2d_3DArray(buffer->SOAin, sizeX, sizeY, current_length+6); */
            GPU::h2d_3DArray(buffer->SOAin, sizeX, sizeY, CHUNK_LENGTH+6);

            ///////////////////////////////////////////////////////////////
            // 4.)
            ///////////////////////////////////////////////////////////////
            for (int i = 0; i < N_chunks; ++i)
                _process_chunk<Kflow, Kupdate>(a, b, dtinvh, src, tmp);

            ///////////////////////////////////////////////////////////////
            // 5.)
            ///////////////////////////////////////////////////////////////
            const uint_t prevOFFSET = SLICE_GPU * previous_iz;

            // GPU rhs into tmp (d2h finishes first for rhs)
            GPU::d2h_rhs_wait();
            timer.start();
            _copy_range(tmp, prevOFFSET, previous_buffer->SOAtmp, 0, SLICE_GPU * previous_length);
            const double t2 = timer.stop();
            if (chatty)
                printf("\t[COPY BACK TMP CHUNK %d TAKES %f sec]\n", previous_chunk_id, t2);

            // GPU update into src (a.k.a updated flow data)
            GPU::d2h_tmp_wait();
            timer.start();
            _copy_range(src, prevOFFSET, previous_buffer->SOAout, 0, SLICE_GPU * previous_length);
            const double t3 = timer.stop();
            if (chatty)
                printf("\t[COPY BACK OUTPUT CHUNK %d TAKES %f sec]\n", previous_chunk_id, t3);

            if (chatty) _end_info_current_chunk();

            return tall.stop();
        }


        // info
        inline uint_t number_of_chunks() const { return N_chunks; }
        inline uint_t chunk_length() const { return current_length; }
        inline uint_t chunk_start_iz() const { return current_iz; }
        inline uint_t chunk_id() const { return current_chunk_id; }
        inline void toggle_verbosity()
        {
            switch (chatty)
            {
                case QUIET:   chatty = VERBOSE; break;
                case VERBOSE: chatty = QUIET; break;
            }
        }
};

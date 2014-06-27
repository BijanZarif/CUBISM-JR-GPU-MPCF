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

#define ID3(x,y,z,NX,NY) ((x) + (NX) * ((y) + (NY) * (z)))


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
        enum {SKIN, FLESH} myFeature[6];

        typedef uint_t (*index_map)(const int ix, const int iy, const int iz);

        // GPU BUFFER SIZES
        const uint_t GPU_input_size;
        const uint_t GPU_output_size;

        // CHUNK RELATED
        const uint_t CHUNK_LENGTH;
        const uint_t REM;
        const uint_t N_chunks;
        uint_t current_length;
        uint_t current_iz;
        uint_t current_chunk_id;
        uint_t previous_length;
        uint_t previous_iz;
        uint_t previous_chunk_id;

        int* maxSOS; // pointer to mapped memory CPU/GPU

        ///////////////////////////////////////////////////////////////////////
        // HALOS / COMMUNICATION
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
        } halox, haloy, haloz;

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
        void _copysend_halos(const int sender, Real * const cpybuf, const uint_t Nhalos, const int xS, const int xE, const int yS, const int yE, const int zS, const int zE)
        {
            assert(Nhalos == (xE-xS)*(yE-yS)*(zE-zS));

#               pragma omp parallel for
            for (int p = 0; p < GridMPI::NVAR; ++p)
            {
                const Real * const src = grid.pdata()[p];
                const uint_t offset = p * Nhalos;
                for (int iz = zS; iz < zE; ++iz)
                    for (int iy = yS; iy < yE; ++iy)
                        for (int ix = xS; ix < xE; ++ix)
                            cpybuf[offset + map(ix,iy,iz)] = src[ix + sizeX * (iy + sizeY * iz)];
            }

            _issue_send(cpybuf, GridMPI::NVAR * Nhalos, sender);
        }

        void _copysend_halos(const int sender, Real * const cpybuf, const uint_t Nhalos, const int zS)
        {
            assert(Nhalos == 3*SLICE_GPU);

            const uint_t srcoffset = SLICE_GPU * zS;
#           pragma omp parallel for
            for (int p = 0; p < GridMPI::NVAR; ++p)
            {
                const Real * const src = grid.pdata()[p];
                const uint_t offset = p * Nhalos;
                memcpy(cpybuf + offset, src + srcoffset, Nhalos*sizeof(Real));
            }

            _issue_send(cpybuf, GridMPI::NVAR * Nhalos, sender);
        }


        ///////////////////////////////////////////////////////////////////////
        // HOST BUFFERS (pinned)
        ///////////////////////////////////////////////////////////////////////
        struct HostBuffer
        {
            static const uint_t NVAR = GridMPI::NVAR; // number of variables in set
            const uint_t _sizeIn, _sizeOut;
            uint_t Nxghost, Nyghost; // may change depending on last chunk
            /* const uint_t Nzghost; // must not change */

            // Tmp storage for GPU input data SoA representation
            cuda_vector_t SOA_all;
            RealPtrVec_t SOA;

            // Tmp storage for GPU tmp SoA representation
            cuda_vector_t tmpSOA_all;
            RealPtrVec_t tmpSOA;

            // compact ghosts
            cuda_vector_t xyghost_all;
            RealPtrVec_t xghost_l, xghost_r, yghost_l, yghost_r; //zghost_l, zghost_r;

            HostBuffer(const uint_t sizeIn, const uint_t sizeOut, const uint_t sizeXghost, const uint_t sizeYghost) :
            /* HostBuffer(const uint_t sizeIn, const uint_t sizeOut, const uint_t sizeXghost, const uint_t sizeYghost, const uint_t sizeZghost) : */
                _sizeIn(sizeIn), _sizeOut(sizeOut),
                Nxghost(sizeXghost), Nyghost(sizeYghost),
                /* Nzghost(sizeZghost), */
                SOA_all(NVAR*sizeIn, 0.0), SOA(NVAR, NULL),
                tmpSOA_all(NVAR*sizeOut, 0.0), tmpSOA(NVAR, NULL),
                xyghost_all(2*NVAR*sizeXghost + 2*NVAR*sizeYghost, 0.0),
                /* ghost_all(2*NVAR*sizeXghost + 2*NVAR*sizeYghost + 2*NVAR*sizeZghost, 0.0), */
                xghost_l(NVAR, NULL), xghost_r(NVAR, NULL),
                yghost_l(NVAR, NULL), yghost_r(NVAR, NULL)
                /* zghost_l(NVAR, NULL), zghost_r(NVAR, NULL) */
            {
                for (uint_t i = 0; i < NVAR; ++i)
                {
                    SOA[i]    = &SOA_all[i * sizeIn];
                    tmpSOA[i] = &tmpSOA_all[i * sizeOut];
                }
                realign_ghost_pointer(sizeXghost, sizeYghost);
                /* realign_ghost_pointer(sizeXghost, sizeYghost, sizeZghost); */
            }

            // --> CURRENTLY NOT USED ELSEWHERE, THEREFORE REQUIRES sizeZ % CHUNK_LENGTH == 0
            // this is dangerous!  Realign the ghost buffer pointers which
            // allows to copy a reduced buffer size to the GPU, e.g. if REM !=
            // 0 for the last chunk in the queue. (Data in the buffers will be
            // coalesced at all times.  Ghosts are stored all in the same
            // contiguous array for better H2D/D2H bandwidth)
            void realign_ghost_pointer(const uint_t sizeXg, const uint_t sizeYg)
            /* void realign_ghost_pointer(const uint_t sizeXg, const uint_t sizeYg, const uint_t sizeZg) */
            {
                assert(sizeXg  <= Nxghost);
                assert(sizeYg  <= Nyghost);
                /* assert(sizeZg  <= Nzghost); */

                const uint_t allXghost = 2*NVAR*sizeXg;
                /* const uint_t allYghost = 2*NVAR*sizeYg; */
                for (uint_t i = 0; i < NVAR; ++i)
                {
                    xghost_l[i] = &xyghost_all[(0*NVAR + i) * sizeXg];
                    xghost_r[i] = &xyghost_all[(1*NVAR + i) * sizeXg];
                    yghost_l[i] = &xyghost_all[(0*NVAR + i) * sizeYg + allXghost];
                    yghost_r[i] = &xyghost_all[(1*NVAR + i) * sizeYg + allXghost];
                    /* zghost_l[i] = &ghost_all[(0*NVAR + i) * sizeZg + allXghost + allYghost]; */
                    /* zghost_r[i] = &ghost_all[(1*NVAR + i) * sizeZg + allXghost + allYghost]; */
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
        void _alloc_GPU()
        {
            GPU::alloc((void**) &maxSOS, sizeX, sizeY, sizeZ, CHUNK_LENGTH);
            gpu_allocation = ALLOCATED;
        }

        void _free_GPU()
        {
            GPU::dealloc();
            gpu_allocation = FREE;
        }

        inline void _syncGPU() { GPU::syncGPU(); }
        inline void _syncStream(GPU::streamID s) { GPU::syncStream(s); }

        void _reset()
        {
            if (N_chunks == 1)
            {
                // whole chunk fits on the GPU
                chunk_state = SINGLE;
                current_length = sizeZ;
            }
            else
            {
                chunk_state = FIRST;
                current_length = CHUNK_LENGTH;
            }
            current_iz = 0;
            current_chunk_id = 1;
        }

        void _init_next_chunk()
        {
            previous_length   = current_length;
            previous_iz       = current_iz;
            previous_chunk_id = current_chunk_id;

            current_iz     += current_length;
            ++current_chunk_id;

            if (current_chunk_id > N_chunks)
                _reset();
            else if (current_chunk_id == N_chunks)
            {
                current_length  = (1 - !REM)*REM + (!REM)*CHUNK_LENGTH;
                chunk_state     = LAST;
            }
            else
            {
                current_length = CHUNK_LENGTH;
                chunk_state    = INTERMEDIATE;
            }

            // use a new host buffer
            _swap_buffer();

            // set the number of ghosts in x/y direction for this buffer
            buffer->Nxghost = 3*sizeY*current_length;
            buffer->Nyghost = sizeX*3*current_length;
        }

        inline void _copy_range(RealPtrVec_t& dst, const uint_t dstOFFSET, const RealPtrVec_t& src, const uint_t srcOFFSET, const uint_t Nelements)
        {
            for (int i = 0; i < GridMPI::NVAR; ++i)
                memcpy(dst[i] + dstOFFSET, src[i] + srcOFFSET, Nelements*sizeof(Real));
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
            GPU::h2d_3DArray(buffer->SOA, sizeX, sizeY, CHUNK_LENGTH);

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
            switch (chunk_state)
            {
                case FIRST: // Pre-computes for rhs coputation after maxSOS
                    // interior + right ghosts
                    _copy_range(buffer->SOA, haloz.Nhalo, src, OFFSET, SLICE_GPU * current_length + haloz.Nhalo);
                    break;

                case SINGLE: // Pre-computes for rhs computation after maxSOS
                    // interior only
                    _copy_range(buffer->SOA, haloz.Nhalo, src, OFFSET, SLICE_GPU * current_length);
                    break;

                default: // Pre-computes next chunk for maxSOS
                    _copy_range(buffer->SOA, 0, src, OFFSET, SLICE_GPU * current_length);
                    break;
            }
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
             * 11.) Upload GPU input of new chunk (3DArrays, overlaps download of 8-9, uploaded on MAIN stream)
             * */

            Timer timer;

            ///////////////////////////////////////////////////////////////////
            // 1.)
            ///////////////////////////////////////////////////////////////////
            uint_t OFFSET = SLICE_GPU * current_iz;
            timer.start();
            _copy_range(buffer->tmpSOA, 0, tmp, OFFSET, SLICE_GPU * current_length);
            const double t1 = timer.stop();
            if (chatty)
                printf("\t[COPY TMP CHUNK %d TAKES %f sec]\n", current_chunk_id, t1);

            ///////////////////////////////////////////////////////////////////
            // 2.)
            ///////////////////////////////////////////////////////////////////
            /* GPU::h2d_tmp(buffer->tmpSOA, SLICE_GPU * current_length); */
            GPU::h2d_tmp(buffer->tmpSOA, GPU_output_size);

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
                // Since operations are on previous chunk, this must only
                // be done on INTERMEDIATE or LAST chunks.  The SOA->AOS
                // copy back of SINGLE and (actual) LAST chunks must be
                // done after _process_chunk has finished.  The next
                // lab.load() operation requires a fully updated domain.
                case INTERMEDIATE:
                case LAST: // operations are on chunk one before LAST!
                    const uint_t prevOFFSET = SLICE_GPU * previous_iz;

                    GPU::d2h_rhs_wait(); // make sure previous d2h has finished
                    timer.start();
                    _copy_range(tmp, prevOFFSET, previous_buffer->tmpSOA, 0, SLICE_GPU * previous_length);
                    const double t4 = timer.stop();
                    if (chatty)
                        printf("\t[COPY BACK TMP CHUNK %d TAKES %f sec]\n", previous_chunk_id, t4);

                    GPU::d2h_tmp_wait();
                    timer.start();
                    _copy_range(src, prevOFFSET, previous_buffer->SOA, 0, SLICE_GPU * previous_length);
                    const double t2 = timer.stop();
                    if (chatty)
                        printf("\t[COPY BACK SRC CHUNK %d TAKES %f sec]\n", previous_chunk_id, t2);
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
                // Prepare interior nodes for the next call to _process_all.
                // The ghosts are copied later, once the new Lab has been
                // loaded.
                case FIRST:
                    // interior + right ghosts
                    _copy_range(buffer->SOA, haloz.Nhalo, src, OFFSET, SLICE_GPU * current_length + haloz.Nhalo);
                    break;

                case INTERMEDIATE:
                    {
                        // left ghosts (reuse conversion in previous buffer)
                        const uint_t prevOFFSET = SLICE_GPU * previous_length;
                        _copy_range(buffer->SOA, 0, previous_buffer->SOA, prevOFFSET, haloz.Nhalo);
                        /* for (int i = 0; i < 7; ++i) */
                        /*     memcpy(buffer->SOA[i], previous_buffer->SOA[i] + prevOFFSET, buffer->Nzghost*sizeof(Real)); */

                        // interior + right ghosts
                        _copy_range(buffer->SOA, haloz.Nhalo, src, OFFSET, SLICE_GPU * current_length + haloz.Nhalo);
                        break;
                    }

                case LAST:
                    {
                        // left ghosts (reuse conversion in previous buffer)
                        const uint_t prevOFFSET = SLICE_GPU * previous_length;
                        _copy_range(buffer->SOA, 0, previous_buffer->SOA, prevOFFSET, haloz.Nhalo);
                        /* for (int i = 0; i < 7; ++i) */
                        /*     memcpy(buffer->SOA[i], previous_buffer->SOA[i] + prevOFFSET, buffer->Nzghost*sizeof(Real)); */

                        // interior
                        _copy_range(buffer->SOA, haloz.Nhalo, src, OFFSET, SLICE_GPU * current_length);

                        // right ghosts
                        /* _compute_z_ghosts_r(src0); */
                        const uint_t current_rightOFFSET = haloz.Nhalo + SLICE_GPU * current_length;
                        _copy_range(buffer->SOA, current_rightOFFSET, haloz.right, 0, haloz.Nhalo);
                        /* for (int i = 0; i < 7; ++i) */
                        /*     memcpy(buffer->SOA[i] + current_rightOFFSET, buffer->zghost_r[i], buffer->Nzghost*sizeof(Real)); */
                        break;
                    }
            }
            const double t3 = timer.stop();
            if (chatty)
                printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", current_chunk_id, t3);

            ///////////////////////////////////////////////////////////////////
            // 8.)
            ///////////////////////////////////////////////////////////////////
            /* GPU::d2h_rhs(previous_buffer->tmpSOA, SLICE_GPU * previous_length); */
            GPU::d2h_rhs(previous_buffer->tmpSOA, GPU_output_size);

            ///////////////////////////////////////////////////////////////////
            // 9.)
            ///////////////////////////////////////////////////////////////////
            /* GPU::d2h_tmp(previous_buffer->SOA, SLICE_GPU * previous_length); */
            GPU::d2h_tmp(previous_buffer->SOA, GPU_output_size);

            ///////////////////////////////////////////////////////////////////
            // 11.)
            ///////////////////////////////////////////////////////////////////
            switch (chunk_state)
            {
                case INTERMEDIATE:
                case LAST:
                    _copy_xyghosts();
                    /* _compute_xy_ghosts(src0); */
                    GPU::upload_xy_ghosts(buffer->Nxghost, buffer->xghost_l, buffer->xghost_r,
                                          buffer->Nyghost, buffer->yghost_l, buffer->yghost_r);

                    /* GPU::h2d_3DArray(buffer->SOA, sizeX, sizeY, current_length+6); */
                    GPU::h2d_3DArray(buffer->SOA, sizeX, sizeY, CHUNK_LENGTH+6);
                    break;
            }
        }

        // info
        void _printSOA(const Real * const in)
        {
            for (int iz = 0; iz < current_length+6; ++iz)
            {
                for (int iy = 0; iy < sizeY; ++iy)
                {
                    for (int ix = 0; ix < sizeX; ++ix)
                        printf("%f\t\n", in[ix + sizeX * (iy + sizeY * iz)]);
                    printf("\n");
                }
                printf("\n");
            }
        }

        void _show_feature();
        void _start_info_current_chunk(const std::string title = "");
        inline void _end_info_current_chunk()
        {
            printf("}\n");
        }


    protected:

        GridMPI& grid;

        virtual void _apply_bc(const double t = 0) {}

        // index mappings
        template <int A, int B, int C>
        static inline uint_t halomap_x(const int ix, const int iy, const int iz)
        {
            return ID3(iy, ix+A, iz, B, C);
        }


        template <int A, int B, int C>
        static inline uint_t halomap_y(const int ix, const int iy, const int iz)
        {
            return ID3(ix, iy+A, iz, B, C);
        }


        template <int A, int B, int C>
        static inline uint_t halomap_z(const int ix, const int iy, const int iz)
        {
            return ID3(ix, iy, iz+A, B, C);
        }


        inline void _copy_xyghosts() // alternatively, copy ALL x/yghosts at beginning
        {
            // copy from the halos into the buffer of the current chunk
            _copy_range(buffer->xghost_l, 0, halox.left,  3*sizeY*current_iz, buffer->Nxghost);
            _copy_range(buffer->xghost_r, 0, halox.right, 3*sizeY*current_iz, buffer->Nxghost);
            _copy_range(buffer->yghost_l, 0, haloy.left,  3*sizeX*current_iz, buffer->Nyghost);
            _copy_range(buffer->yghost_r, 0, haloy.right, 3*sizeX*current_iz, buffer->Nyghost);
        }


        /* void _compute_xy_ghosts(const Real * const src0) */
        /* { */
        /*     // Compute the x- and yghost for the current chunk */
        /*     const int stencilStart[3] = {-3, -3, -3}; */
        /*     const int stencilEnd[3]   = { 4,  4,  4}; */

        /*     /1* // use a boundary condition applied to the (current chunk) *1/ */
        /*     /1* BoundaryCondition_CUDA<FluidBlock, FluidBlock::ElementType> bc(stencilStart, stencilEnd, src0, current_iz, current_length); *1/ */
        /*     /1* bc.template applyBC_absorbing<0,0>(buffer->xghost_l, &GPUlab::idx_xghosts< 3,                  FluidBlock::sizeY, 3>); *1/ */
        /*     /1* bc.template applyBC_absorbing<0,1>(buffer->xghost_r, &GPUlab::idx_xghosts< -FluidBlock::sizeX, FluidBlock::sizeY, 3>); *1/ */
        /*     /1* bc.template applyBC_absorbing<1,0>(buffer->yghost_l, &GPUlab::idx_yghosts< 3,                  FluidBlock::sizeX, 3>); *1/ */
        /*     /1* bc.template applyBC_absorbing<1,1>(buffer->yghost_r, &GPUlab::idx_yghosts< -FluidBlock::sizeY, FluidBlock::sizeX, 3>); *1/ */
        /* } */


        /* void _compute_z_ghosts_l(const Real * const src0) */
        /* { */
        /*     // Compute the left zghost for the current chunk */
        /*     const int stencilStart[3] = {-3, -3, -3}; */
        /*     const int stencilEnd[3]   = { 4,  4,  4}; */

        /*     /1* BoundaryCondition_CUDA<FluidBlock, FluidBlock::ElementType> bc(stencilStart, stencilEnd, src0); *1/ */
        /*     /1* bc.template applyBC_absorbing<2,0>(buffer->zghost_l, &GPUlab::idx_zghosts< 3, FluidBlock::sizeX, FluidBlock::sizeY>); *1/ */
        /* } */


        /* void _compute_z_ghosts_r(const Real * const src0) */
        /* { */
        /*     // Compute the right zghost for the current chunk */
        /*     const int stencilStart[3] = {-3, -3, -3}; */
        /*     const int stencilEnd[3]   = { 4,  4,  4}; */

        /*     /1* BoundaryCondition_CUDA<FluidBlock, FluidBlock::ElementType> bc(stencilStart, stencilEnd, src0); *1/ */
        /*     /1* bc.template applyBC_absorbing<2,1>(buffer->zghost_r, &GPUlab::idx_zghosts< -FluidBlock::sizeZ, FluidBlock::sizeX, FluidBlock::sizeY>); *1/ */
        /* } */


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
            _copy_range(buffer->SOA, 0, src, 0, SLICE_GPU * current_length);
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
             * 2.) Copy left (and right for SINGLE) ghosts (hidden by 1.)
             * 3.) Upload GPU input for FIRST/SINGLE chunk (3DArrays)
             * 4.) Process all chunks
             * 5.) SOA->AOS of GPU updated solution for LAST/SINGLE chunk
             * 6.) If chunk is SINGLE: AOS->SOA of interior nodes (can not be hidden for SINGLE case)
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
            _copy_xyghosts();
            /* _compute_xy_ghosts(src0); */
            GPU::upload_xy_ghosts(buffer->Nxghost, buffer->xghost_l, buffer->xghost_r,
                                  buffer->Nyghost, buffer->yghost_l, buffer->yghost_r);

            ///////////////////////////////////////////////////////////////
            // 2.)
            ///////////////////////////////////////////////////////////////
            // copy left ghosts always (CAN BE DONE BE MPI RECV)
            _copy_range(buffer->SOA, 0, haloz.left, 0, haloz.Nhalo);
            /* _copy_range(buffer->SOA, 0, buffer->zghost_l, 0, buffer->Nzghost * sizeof(Real)); */

            /* for (int i = 0; i < 7; ++i) */
            /*     memcpy(buffer->SOA[i], buffer->zghost_l[i], buffer->Nzghost*sizeof(Real)); */

            // right ghosts are conditional
            switch (chunk_state)
            {
                case SINGLE:
                    /* _compute_z_ghosts_r(src0); */
                    const uint_t OFFSET = haloz.Nhalo + SLICE_GPU * current_length;
                    _copy_range(buffer->SOA, OFFSET, haloz.right, 0, haloz.Nhalo);
                    /* for (int i = 0; i < 7; ++i) */
                    /*     memcpy(buffer->SOA[i] + OFFSET, buffer->zghost_r[i], buffer->Nzghost*sizeof(Real)); */
                    break;
            }

            ///////////////////////////////////////////////////////////////
            // 3.)
            ///////////////////////////////////////////////////////////////
            /* GPU::h2d_3DArray(buffer->SOA, sizeX, sizeY, current_length+6); */
            GPU::h2d_3DArray(buffer->SOA, sizeX, sizeY, CHUNK_LENGTH+6);

            ///////////////////////////////////////////////////////////////
            // 4.)
            ///////////////////////////////////////////////////////////////
            for (int i = 0; i < N_chunks; ++i)
                _process_chunk<Kflow, Kupdate>(a, b, dtinvh, src, tmp);

            ///////////////////////////////////////////////////////////////
            // 5.)
            ///////////////////////////////////////////////////////////////
            // Copy back SOA->AOS of previous chunk
            Timer timer;
            const uint_t prevOFFSET = SLICE_GPU * previous_iz;

            // GPU rhs into tmp (d2h finishes first for rhs)
            GPU::d2h_rhs_wait();
            timer.start();
            _copy_range(tmp, prevOFFSET, previous_buffer->tmpSOA, 0, SLICE_GPU * previous_length);
            const double t4 = timer.stop();
            if (chatty)
                printf("\t[COPY BACK TMP CHUNK %d TAKES %f sec]\n", previous_chunk_id, t4);

            // GPU update into src (a.k.a updated flow data)
            GPU::d2h_tmp_wait();
            timer.start();
            _copy_range(src, prevOFFSET, previous_buffer->SOA, 0, SLICE_GPU * previous_length);
            const double t1 = timer.stop();
            if (chatty)
                printf("\t[COPY BACK SRC CHUNK %d TAKES %f sec]\n", previous_chunk_id, t1);

            ///////////////////////////////////////////////////////////////
            // 6.)
            ///////////////////////////////////////////////////////////////
            switch (chunk_state)
            {
                case SINGLE:
                    timer.start();
                    _copy_range(buffer->SOA, haloz.Nhalo, src, 0, SLICE_GPU * current_length);
                    const double t3 = timer.stop();
                    if (chatty)
                        printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", current_chunk_id, t3);
                    break;
            }
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

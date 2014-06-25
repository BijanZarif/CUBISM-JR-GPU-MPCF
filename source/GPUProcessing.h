/* *
 * GPUProcessing.h
 *
 * Created by Fabian Wermelinger on 5/28/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include "GPU.h"
/* #include "BoundaryConditions_CUDA.h" */
#include "Types.h"
#include "Timer.h"

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <string>


#ifndef _PAGEABLE_HOST_MEM_
#include "cudaHostAllocator.h"
typedef std::vector<Real, cudaHostAllocator<Real> > cuda_vector_t;
#else
typedef std::vector<Real> cuda_vector_t;
#endif


class GPUProcessing
{
    private:

        // Domain size (assume cube)
        const uint_t BSX_GPU, BSY_GPU, BSZ_GPU;
        const uint_t CHUNK_LENGTH;
        const uint_t SLICE_GPU;
        const uint_t REM;

        // GPU chunk
        const uint_t N_chunks;
        const uint_t GPU_input_size;
        const uint_t GPU_output_size;
        uint_t current_length;
        uint_t current_iz;
        uint_t current_chunk_id;
        uint_t previous_length;
        uint_t previous_iz;
        uint_t previous_chunk_id;

        // Ghosts
        const uint_t N_xyghosts;
        const uint_t N_zghosts;

        // Max SOS
        int* maxSOS;

        // SOA data buffers (pinned)
        struct SOAdata
        {
            // Tmp storage for GPU input data SoA representation
            cuda_vector_t SOA_all;
            RealPtrVec_t SOA;

            // Tmp storage for GPU tmp SoA representation
            cuda_vector_t tmpSOA_all;
            RealPtrVec_t tmpSOA;

            // compact ghosts
            cuda_vector_t ghost_all;
            RealPtrVec_t xghost_l, xghost_r, yghost_l, yghost_r, zghost_l, zghost_r;

            SOAdata(const uint_t sizeIn, const uint_t sizeOut, const uint_t sizeXYghost, const uint_t sizeZghost) :
                SOA_all(7*sizeIn, 0.0), SOA(7, NULL),
                tmpSOA_all(7*sizeOut, 0.0), tmpSOA(7, NULL),
                ghost_all(4*7*sizeXYghost + 2*7*sizeZghost, 0.0),
                xghost_l(7, NULL), xghost_r(7, NULL),
                yghost_l(7, NULL), yghost_r(7, NULL),
                zghost_l(7, NULL), zghost_r(7, NULL)
            {
                for (uint_t i = 0; i < 7; ++i)
                {
                    SOA[i]    = &SOA_all[i * sizeIn];
                    tmpSOA[i] = &tmpSOA_all[i * sizeOut];
                    xghost_l[i] = &ghost_all[(0*7 + i) * sizeXYghost];
                    xghost_r[i] = &ghost_all[(1*7 + i) * sizeXYghost];
                    yghost_l[i] = &ghost_all[(2*7 + i) * sizeXYghost];
                    yghost_r[i] = &ghost_all[(3*7 + i) * sizeXYghost];
                    zghost_l[i] = &ghost_all[(0*7 + i) * sizeZghost + 4*7*sizeXYghost];
                    zghost_r[i] = &ghost_all[(1*7 + i) * sizeZghost + 4*7*sizeXYghost];
                }
            }
        };

        // using 2 SOA buffers for additional CPU overlap
        SOAdata BUFFER1, BUFFER2;
        SOAdata *buffer; // active buffer
        SOAdata *previous_buffer;
        inline void _switch_buffer() // switch active buffer
        {
            previous_buffer = buffer;
            if (buffer == &BUFFER1)
                buffer = &BUFFER2;
            else
                buffer = &BUFFER1;
        }

        // states
        enum {FIRST, INTERMEDIATE, LAST, SINGLE} chunk_state;
        enum {ALLOCATED, FREE} gpu_allocation;
        enum {QUIET=0, VERBOSE} chatty;

        // helpers
        void _alloc_GPU();
        void _free_GPU();
        void _reset();
        void _init_next_chunk();
        inline void _copy_chunk(RealPtrVec_t& dst, const uint_t dstOFFSET, const RealPtrVec_t& src, const uint_t srcOFFSET, const uint_t Nelements)
        {
            for (int i = 0; i < 7; ++i)
                memcpy(dst[i] + dstOFFSET, src[i] + srcOFFSET, Nelements*sizeof(Real));
        }

        // info
        void _printSOA(const Real * const in);
        void _start_info_current_chunk(const std::string title = "");
        inline void _end_info_current_chunk()
        {
            printf("}\n");
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
                GPU::h2d_3DArray(buffer->SOA, BSX_GPU, BSY_GPU, current_length);

                ///////////////////////////////////////////////////////////////////
                // 2.)
                ///////////////////////////////////////////////////////////////////
                Ksos kernel;
                if (chatty) printf("\t[LAUNCH SOS KERNEL CHUNK %d]\n", current_chunk_id);
                kernel.compute(BSX_GPU, BSY_GPU, current_length);
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
                        _copy_chunk(buffer->SOA, N_zghosts, src, OFFSET, SLICE_GPU * current_length + N_zghosts);
                        break;

                    case SINGLE: // Pre-computes for rhs computation after maxSOS
                        // interior only
                        _copy_chunk(buffer->SOA, N_zghosts, src, OFFSET, SLICE_GPU * current_length);
                        break;

                    default: // Pre-computes next chunk for maxSOS
                        _copy_chunk(buffer->SOA, 0, src, OFFSET, SLICE_GPU * current_length);
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
                _copy_chunk(buffer->tmpSOA, 0, tmp, OFFSET, SLICE_GPU * current_length);
                const double t1 = timer.stop();
                if (chatty)
                    printf("\t[COPY TMP CHUNK %d TAKES %f sec]\n", current_chunk_id, t1);

                ///////////////////////////////////////////////////////////////////
                // 2.)
                ///////////////////////////////////////////////////////////////////
                GPU::h2d_tmp(buffer->tmpSOA, SLICE_GPU * current_length);

                ///////////////////////////////////////////////////////////////////
                // 3.)
                ///////////////////////////////////////////////////////////////////
                Kflow convection(a, dtinvh);
                if (chatty) printf("\t[LAUNCH CONVECTION KERNEL CHUNK %d]\n", current_chunk_id);
                /* convection.compute(BSX_GPU, BSY_GPU, current_length, current_iz); */
                convection.compute(BSX_GPU, BSY_GPU, current_length, 0);

                ///////////////////////////////////////////////////////////////////
                // 4.)
                ///////////////////////////////////////////////////////////////////
                Kupdate update(b);
                if (chatty) printf("\t[LAUNCH UPDATE KERNEL CHUNK %d]\n", current_chunk_id);
                update.compute(BSX_GPU, BSY_GPU, current_length);

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
                        _copy_chunk(tmp, prevOFFSET, previous_buffer->tmpSOA, 0, SLICE_GPU * previous_length);
                        const double t4 = timer.stop();
                        if (chatty)
                            printf("\t[COPY BACK TMP CHUNK %d TAKES %f sec]\n", previous_chunk_id, t4);

                        GPU::d2h_tmp_wait();
                        timer.start();
                        _copy_chunk(src, prevOFFSET, previous_buffer->SOA, 0, SLICE_GPU * previous_length);
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
                /* RealPtrVec_t first_interior(7); // pointer to first interior node */
                /* for (int i = 0; i < 7; ++i) */
                /*     first_interior[i] = buffer->SOA[i] + N_zghosts; */

                OFFSET = SLICE_GPU * current_iz;

                timer.start();
                switch (chunk_state)
                {
                    // Prepare interior nodes for the next call to _process_all.
                    // The ghosts are copied later, once the new Lab has been
                    // loaded.
                    case FIRST:
                        // interior + right ghosts
                        _copy_chunk(buffer->SOA, N_zghosts, src, OFFSET, SLICE_GPU * current_length + N_zghosts);
                        break;

                    case INTERMEDIATE:
                        {
                            // left ghosts (reuse conversion in previous buffer)
                            const uint_t prevOFFSET = SLICE_GPU * previous_length;
                            for (int i = 0; i < 7; ++i)
                                memcpy(buffer->SOA[i], previous_buffer->SOA[i] + prevOFFSET, N_zghosts*sizeof(Real));

                            // interior + right ghosts
                            _copy_chunk(buffer->SOA, N_zghosts, src, OFFSET, SLICE_GPU * current_length + N_zghosts);
                            break;
                        }

                    case LAST:
                        {
                            // left ghosts (reuse conversion in previous buffer)
                            const uint_t prevOFFSET = SLICE_GPU * previous_length;
                            for (int i = 0; i < 7; ++i)
                                memcpy(buffer->SOA[i], previous_buffer->SOA[i] + prevOFFSET, N_zghosts*sizeof(Real));

                            // interior
                            _copy_chunk(buffer->SOA, N_zghosts, src, OFFSET, SLICE_GPU * current_length);

                            // right ghosts
                            /* _compute_z_ghosts_r(src0); */
                            const uint_t current_rightOFFSET = N_zghosts + SLICE_GPU * current_length;
                            for (int i = 0; i < 7; ++i)
                                memcpy(buffer->SOA[i] + current_rightOFFSET, buffer->zghost_r[i], N_zghosts*sizeof(Real));
                            break;
                        }
                }
                const double t3 = timer.stop();
                if (chatty)
                    printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", current_chunk_id, t3);

                ///////////////////////////////////////////////////////////////////
                // 8.)
                ///////////////////////////////////////////////////////////////////
                GPU::d2h_rhs(previous_buffer->tmpSOA, SLICE_GPU * previous_length);

                ///////////////////////////////////////////////////////////////////
                // 9.)
                ///////////////////////////////////////////////////////////////////
                GPU::d2h_tmp(previous_buffer->SOA, SLICE_GPU * previous_length);

                ///////////////////////////////////////////////////////////////////
                // 11.)
                ///////////////////////////////////////////////////////////////////
                switch (chunk_state)
                {
                    case INTERMEDIATE:
                    case LAST:
                        /* _compute_xy_ghosts(src0); */
                        GPU::upload_xy_ghosts(N_xyghosts,
                                buffer->xghost_l, buffer->xghost_r,
                                buffer->yghost_l, buffer->yghost_r);

                        GPU::h2d_3DArray(buffer->SOA, BSX_GPU, BSY_GPU, current_length+6);
                        break;
                }
            }


        /* // index computing functions to index into the ghost buffers and generate */
        /* // the correct stride for the GPU */
        /* template <int A, int B, int C> */
        /*     static inline uint_t idx_xghosts(const int ix, const int iy, const int iz) */
        /*     { */
        /*         return ID3(iy, ix+A, iz, B, C); */
        /*     } */


        /* template <int A, int B, int C> */
        /*     static inline uint_t idx_yghosts(const int ix, const int iy, const int iz) */
        /*     { */
        /*         return ID3(ix, iy+A, iz, B, C); */
        /*     } */


        /* template <int A, int B, int C> */
        /*     static inline uint_t idx_zghosts(const int ix, const int iy, const int iz) */
        /*     { */
        /*         return ID3(ix, iy, iz+A, B, C); */
        /*     } */


        void _compute_xy_ghosts(const Real * const src0)
        {
            // Compute the x- and yghost for the current chunk
            const int stencilStart[3] = {-3, -3, -3};
            const int stencilEnd[3]   = { 4,  4,  4};

            /* // use a boundary condition applied to the (current chunk) */
            /* BoundaryCondition_CUDA<FluidBlock, FluidBlock::ElementType> bc(stencilStart, stencilEnd, src0, current_iz, current_length); */
            /* bc.template applyBC_absorbing<0,0>(buffer->xghost_l, &GPUProcessing::idx_xghosts< 3,                  FluidBlock::sizeY, 3>); */
            /* bc.template applyBC_absorbing<0,1>(buffer->xghost_r, &GPUProcessing::idx_xghosts< -FluidBlock::sizeX, FluidBlock::sizeY, 3>); */
            /* bc.template applyBC_absorbing<1,0>(buffer->yghost_l, &GPUProcessing::idx_yghosts< 3,                  FluidBlock::sizeX, 3>); */
            /* bc.template applyBC_absorbing<1,1>(buffer->yghost_r, &GPUProcessing::idx_yghosts< -FluidBlock::sizeY, FluidBlock::sizeX, 3>); */
        }


        void _compute_z_ghosts_l(const Real * const src0)
        {
            // Compute the left zghost for the current chunk
            const int stencilStart[3] = {-3, -3, -3};
            const int stencilEnd[3]   = { 4,  4,  4};

            /* BoundaryCondition_CUDA<FluidBlock, FluidBlock::ElementType> bc(stencilStart, stencilEnd, src0); */
            /* bc.template applyBC_absorbing<2,0>(buffer->zghost_l, &GPUProcessing::idx_zghosts< 3, FluidBlock::sizeX, FluidBlock::sizeY>); */
        }


        void _compute_z_ghosts_r(const Real * const src0)
        {
            // Compute the right zghost for the current chunk
            const int stencilStart[3] = {-3, -3, -3};
            const int stencilEnd[3]   = { 4,  4,  4};

            /* BoundaryCondition_CUDA<FluidBlock, FluidBlock::ElementType> bc(stencilStart, stencilEnd, src0); */
            /* bc.template applyBC_absorbing<2,1>(buffer->zghost_r, &GPUProcessing::idx_zghosts< -FluidBlock::sizeZ, FluidBlock::sizeX, FluidBlock::sizeY>); */
        }


    public:

        GPUProcessing(const uint_t BSX, const uint_t BSY, const uint_t BSZ, const uint_t CW);
        ~GPUProcessing();

        // main accessors to process complete domain
        // ========================================================================
        template <typename Ksos>
            double max_sos(const RealPtrVec_t& src, float& sos)
            {
                /* *
                 * 1.) Init maxSOS = 0 (mapped integer -> zero-copy)
                 * 2.) Convert AOS->SOA for the FIRST/SINGLE chunk
                 * 3.) Process all chunks
                 * 4.) Synchronize stream to make sure reduction is complete
                 * */

                Timer tsos;
                tsos.start();

                ///////////////////////////////////////////////////////////////////
                // 1.)
                ///////////////////////////////////////////////////////////////////
                *maxSOS = 0;

                ///////////////////////////////////////////////////////////////////
                // 2.)
                ///////////////////////////////////////////////////////////////////
                Timer timer;
                timer.start();
                _copy_chunk(buffer->SOA, 0, src, 0, SLICE_GPU * current_length);
                const double t1 = timer.stop();
                if (chatty)
                {
                    char title[256];
                    sprintf(title, "MAX SOS PROCESSING CHUNK %d\n", current_chunk_id);
                    _start_info_current_chunk(title);
                    printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", current_chunk_id, t1);
                }

                ///////////////////////////////////////////////////////////////////
                // 3.)
                ///////////////////////////////////////////////////////////////////
                for (int i = 0; i < N_chunks; ++i)
                    _process_chunk<Ksos>(src);
                if (chatty) _end_info_current_chunk();

                ///////////////////////////////////////////////////////////////////
                // 4.)
                ///////////////////////////////////////////////////////////////////
                this->syncStream(GPU::streamID::S1);

                union {float f; int i;} ret;
                ret.i = *maxSOS;
                sos   = ret.f;

                return tsos.stop();
            }


        template <typename Kflow, typename Kupdate>
            double process_all(const Real a, const Real b, const Real dtinvh, RealPtrVec_t& src, RealPtrVec_t& tmp)
            {
                /* *
                 * 1.) Extract x/yghosts for current chunk and upload to GPU
                 * 2.) Copy left (and right for SINGLE) ghosts (hidden by 1.)
                 * 3.) Upload GPU input for FIRST/SINGLE chunk (3DArrays)
                 * 4.) Process all chunks
                 * 5.) SOA->AOS of GPU updated solution for LAST/SINGLE chunk
                 * 6.) If chunk is SINGLE: AOS->SOA of interior nodes (can not be hidden for SINGLE case)
                 * */

                Timer tall;
                tall.start();

                if (chatty)
                {
                    char title[256];
                    sprintf(title, "RHS PROCESSING CHUNK %d\n", current_chunk_id);
                    _start_info_current_chunk(title);
                }

                ///////////////////////////////////////////////////////////////////
                // 1.)
                ///////////////////////////////////////////////////////////////////
                /* _compute_xy_ghosts(src0); */
                GPU::upload_xy_ghosts(N_xyghosts,
                        buffer->xghost_l, buffer->xghost_r,
                        buffer->yghost_l, buffer->yghost_r);

                ///////////////////////////////////////////////////////////////////
                // 2.)
                ///////////////////////////////////////////////////////////////////
                // copy left ghosts always
                /* _compute_z_ghosts_l(src0); */
                for (int i = 0; i < 7; ++i)
                    memcpy(buffer->SOA[i], buffer->zghost_l[i], N_zghosts*sizeof(Real));

                // right ghosts are conditional
                switch (chunk_state)
                {
                    case SINGLE:
                        /* _compute_z_ghosts_r(src0); */
                        const uint_t OFFSET = N_zghosts + SLICE_GPU * current_length;
                        for (int i = 0; i < 7; ++i)
                            memcpy(buffer->SOA[i] + OFFSET, buffer->zghost_r[i], N_zghosts*sizeof(Real));
                        break;
                }

                ///////////////////////////////////////////////////////////////////
                // 3.)
                ///////////////////////////////////////////////////////////////////
                GPU::h2d_3DArray(buffer->SOA, BSX_GPU, BSY_GPU, current_length+6);

                ///////////////////////////////////////////////////////////////////
                // 4.)
                ///////////////////////////////////////////////////////////////////
                for (int i = 0; i < N_chunks; ++i)
                    _process_chunk<Kflow, Kupdate>(a, b, dtinvh, src, tmp);

                ///////////////////////////////////////////////////////////////////
                // 5.)
                ///////////////////////////////////////////////////////////////////
                // Copy back SOA->AOS of previous chunk
                Timer timer;
                const uint_t prevOFFSET = SLICE_GPU * previous_iz;

                // GPU rhs into tmp (d2h finishes first for rhs)
                GPU::d2h_rhs_wait();
                timer.start();
                _copy_chunk(tmp, prevOFFSET, previous_buffer->tmpSOA, 0, SLICE_GPU * previous_length);
                const double t4 = timer.stop();
                if (chatty)
                    printf("\t[COPY BACK TMP CHUNK %d TAKES %f sec]\n", previous_chunk_id, t4);

                // GPU update into src (a.k.a updated flow data)
                GPU::d2h_tmp_wait();
                timer.start();
                _copy_chunk(src, prevOFFSET, previous_buffer->SOA, 0, SLICE_GPU * previous_length);
                const double t1 = timer.stop();
                if (chatty)
                    printf("\t[COPY BACK SRC CHUNK %d TAKES %f sec]\n", previous_chunk_id, t1);

                ///////////////////////////////////////////////////////////////////
                // 6.)
                ///////////////////////////////////////////////////////////////////
                switch (chunk_state)
                {
                    case SINGLE:
                        /* RealPtrVec_t first_interior(7); // pointer to first interior node */
                        /* for (int i = 0; i < 7; ++i) */
                        /*     first_interior[i] = buffer->SOA[i] + N_zghosts; */

                        timer.start();
                        _copy_chunk(buffer->SOA, N_zghosts, src, 0, SLICE_GPU * current_length);
                        const double t3 = timer.stop();
                        if (chatty)
                            printf("\t[COPY SRC CHUNK %d TAKES %f sec]\n", current_chunk_id, t3);
                        break;
                }
                if (chatty) _end_info_current_chunk();

                return tall.stop();
            }

        // synchronize
        inline void syncGPU() { GPU::syncGPU(); }
        inline void syncStream(GPU::streamID s) { GPU::syncStream(s); }

        // info
        inline uint_t number_of_chunks() const { return N_chunks; }
        inline uint_t chunk_length() const { return current_length; }
        inline uint_t chunk_pos() const { return current_iz; }
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

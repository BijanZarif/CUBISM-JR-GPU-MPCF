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
        void _copy_AOS_to_SOA(RealPtrVec_t& dst, const Real * const src, const uint_t gptfloats, const uint_t Nelements);
        void _copy_SOA_to_AOS(Real * const dst, const RealPtrVec_t& src, const uint_t gptfloats, const uint_t Nelements);
        void _init_next_subdomain();

        // info
        void _printSOA(const Real * const in);
        void _info_current_chunk();

        // execution helper
        template <typename Ksos>
            void _process_subdomain(const Real * const src0, const uint_t gptfloats)
            {
                /* *
                 * Processes a subdomain for the maxSOS computation:
                 * 1.) upload active buffer
                 * 2.) launch GPU maxSOS kernel
                 * 3.) ============== Initialize next subdomain ===================
                 * 4.) convert AOS->SOA data for new subdomain
                 * 5.) wait until upload of active buffer has finished
                 * */

                Timer timer;
                if (chatty)
                    _info_current_chunk();

                ///////////////////////////////////////////////////////////////////
                // 1.)
                ///////////////////////////////////////////////////////////////////
                GPU::h2d_3DArray(buffer->SOA, BSX_GPU, BSY_GPU, current_length);

                ///////////////////////////////////////////////////////////////////
                // 2.)
                ///////////////////////////////////////////////////////////////////
                Ksos kernel;
                if (chatty) printf("MaxSpeedOfSound_CUDA Subdomain %d\n", current_chunk_id);
                kernel.compute(BSX_GPU, BSY_GPU, current_length);

                ///////////////////////////////////////////////////////////////////
                // 3.)
                ///////////////////////////////////////////////////////////////////
                _init_next_subdomain();

                ///////////////////////////////////////////////////////////////////
                // 4.)
                ///////////////////////////////////////////////////////////////////
                const uint_t OFFSET = SLICE_GPU * current_iz * gptfloats;
                timer.start();
                switch (chunk_state)
                {
                    case FIRST: // Pre-computes for rhs coputation after maxSOS
                        {
                            // interior + right ghosts
                            RealPtrVec_t first_interior(7);
                            for (int i = 0; i < 7; ++i)
                                first_interior[i] = buffer->SOA[i] + N_zghosts;
                            _copy_AOS_to_SOA(first_interior, src0 + OFFSET, gptfloats, SLICE_GPU * current_length + N_zghosts);
                            break;
                        }

                    case SINGLE: // Pre-computes for rhs computation after maxSOS
                        {
                            // interior only
                            RealPtrVec_t first_interior(7);
                            for (int i = 0; i < 7; ++i)
                                first_interior[i] = buffer->SOA[i] + N_zghosts;
                            _copy_AOS_to_SOA(first_interior, src0 + OFFSET, gptfloats, SLICE_GPU * current_length);
                            break;
                        }

                    default: // Pre-computes next subdomain for maxSOS
                        _copy_AOS_to_SOA(buffer->SOA, src0 + OFFSET, gptfloats, SLICE_GPU * current_length);
                        break;
                }
                const double t1 = timer.stop();
                if (chatty)
                    printf("[AOS->SOA SUBDOMAIN %d TAKES %f sec]\n", current_chunk_id, t1);

                ///////////////////////////////////////////////////////////////////
                // 5.)
                ///////////////////////////////////////////////////////////////////
                GPU::h2d_3DArray_wait();
            }


        template <typename Kflow, typename Kupdate>
            void _process_subdomain(const Real a, const Real b, const Real dtinvh, Real * const src0, Real * const tmp0, const uint_t gptfloats)
            {
                /* *
                 * Process subdomain for the RHS computation:
                 * 1.)  Convert AOS->SOA tmp (hidden by h2d_3DArray)
                 * 2.)  Upload GPU tmp (needed for divergence, uploaded on TMP stream)
                 * 3.)  Launch GPU convection kernel
                 * 4.)  Launch GPU update kernel
                 * 5.)  Convert SOA->AOS of rhs and updated solution (applies to previous subdomain, hidden by 2-4)
                 * 6.)  ============== Initialize next subdomain ===================
                 * 7.)  Convert AOS->SOA for GPU input of new subdomain (hidden by 2-4)
                 * 8.)  Download GPU rhs of previous subdomain into previous buffer (downloaded on TMP stream)
                 * 9.)  Download GPU updated solution of previous subdomain into previous buffer (downloaded on TMP stream)
                 * 10.) Compute x/yghosts of new subdomain and upload to GPU (overlaps download of 8-9, uploaded on MAIN stream)
                 * 11.) Upload GPU input of new subdomain (3DArrays, overlaps download of 8-9, uploaded on MAIN stream)
                 * */

                Timer timer;
                if (chatty)
                    _info_current_chunk();

                ///////////////////////////////////////////////////////////////////
                // 1.)
                ///////////////////////////////////////////////////////////////////
                uint_t OFFSET = SLICE_GPU * current_iz * gptfloats;
                timer.start();
                _copy_AOS_to_SOA(buffer->tmpSOA, tmp0 + OFFSET, gptfloats, SLICE_GPU * current_length);
                const double t1 = timer.stop();
                if (chatty)
                    printf("[tmpAOS->tmpSOA SUBDOMAIN %d TAKES %f sec]\n", current_chunk_id, t1);

                ///////////////////////////////////////////////////////////////////
                // 2.)
                ///////////////////////////////////////////////////////////////////
                GPU::h2d_tmp(buffer->tmpSOA, SLICE_GPU * current_length);

                ///////////////////////////////////////////////////////////////////
                // 3.)
                ///////////////////////////////////////////////////////////////////
                Kflow convection(a, dtinvh);
                if (chatty) printf("Convection_CUDA Subdomain %d\n", current_chunk_id);
                /* convection.compute(BSX_GPU, BSY_GPU, current_length, current_iz); */
                convection.compute(BSX_GPU, BSY_GPU, current_length, 0);

                ///////////////////////////////////////////////////////////////////
                // 4.)
                ///////////////////////////////////////////////////////////////////
                Kupdate update(b);
                if (chatty) printf("Update_CUDA Subdomain %d\n", current_chunk_id);
                update.compute(BSX_GPU, BSY_GPU, current_length);

                ///////////////////////////////////////////////////////////////////
                // 5.)
                ///////////////////////////////////////////////////////////////////
                switch (chunk_state)
                {
                    // Since operations are on previous subdomain, this must only
                    // be done on INTERMEDIATE or LAST subdomains.  The SOA->AOS
                    // copy back of SINGLE and (actual) LAST subdomains must be
                    // done after _process_subdomain has finished.  The next
                    // lab.load() operation requires a fully updated domain.
                    case INTERMEDIATE:
                    case LAST: // operations are on subdomain one before LAST!
                        const uint_t prevOFFSET = SLICE_GPU * previous_iz * gptfloats;

                        GPU::d2h_rhs_wait(); // make sure previous d2h has finished
                        timer.start();
                        _copy_SOA_to_AOS(tmp0 + prevOFFSET, previous_buffer->tmpSOA, gptfloats, SLICE_GPU * previous_length);
                        const double t4 = timer.stop();
                        if (chatty)
                            printf("[tmpSOA->tmpAOS OF SUBDOMAIN %d TAKES %f sec]\n", previous_chunk_id, t4);

                        GPU::d2h_tmp_wait();
                        timer.start();
                        _copy_SOA_to_AOS(src0 + prevOFFSET, previous_buffer->SOA, gptfloats, SLICE_GPU * previous_length);
                        const double t2 = timer.stop();
                        if (chatty)
                            printf("[SOA->AOS OF SUBDOMAIN %d TAKES %f sec]\n", previous_chunk_id, t2);
                }

                ///////////////////////////////////////////////////////////////////
                // 6.)
                ///////////////////////////////////////////////////////////////////
                _init_next_subdomain();

                ///////////////////////////////////////////////////////////////////
                // 7.)
                ///////////////////////////////////////////////////////////////////
                RealPtrVec_t first_interior(7); // pointer to first interior node
                for (int i = 0; i < 7; ++i)
                    first_interior[i] = buffer->SOA[i] + N_zghosts;

                OFFSET = SLICE_GPU * current_iz * gptfloats;

                timer.start();
                switch (chunk_state)
                {
                    // Prepare interior nodes for the next call to _process_all.
                    // The ghosts are copied later, once the new Lab has been
                    // loaded.
                    case FIRST:
                        // interior + right ghosts
                        _copy_AOS_to_SOA(first_interior, src0 + OFFSET, gptfloats, SLICE_GPU * current_length + N_zghosts);
                        break;

                    case INTERMEDIATE:
                        {
                            // left ghosts (reuse conversion in previous buffer)
                            const uint_t prevOFFSET = SLICE_GPU * previous_length;
                            for (int i = 0; i < 7; ++i)
                                memcpy(buffer->SOA[i], previous_buffer->SOA[i] + prevOFFSET, N_zghosts*sizeof(Real));

                            // interior + right ghosts
                            _copy_AOS_to_SOA(first_interior, src0 + OFFSET, gptfloats, SLICE_GPU * current_length + N_zghosts);
                            break;
                        }

                    case LAST:
                        {
                            // left ghosts (reuse conversion in previous buffer)
                            const uint_t prevOFFSET = SLICE_GPU * previous_length;
                            for (int i = 0; i < 7; ++i)
                                memcpy(buffer->SOA[i], previous_buffer->SOA[i] + prevOFFSET, N_zghosts*sizeof(Real));

                            // interior
                            _copy_AOS_to_SOA(first_interior, src0 + OFFSET, gptfloats, SLICE_GPU * current_length);

                            // right ghosts
                            _compute_z_ghosts_r(src0);
                            const uint_t current_rightOFFSET = N_zghosts + SLICE_GPU * current_length;
                            for (int i = 0; i < 7; ++i)
                                memcpy(buffer->SOA[i] + current_rightOFFSET, buffer->zghost_r[i], N_zghosts*sizeof(Real));
                            break;
                        }
                }
                const double t3 = timer.stop();
                if (chatty)
                    printf("[AOS->SOA SUBDOMAIN %d TAKES %f sec]\n", current_chunk_id, t3);

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
                        _compute_xy_ghosts(src0);
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
            // Compute the x- and yghost for the current subdomain
            const int stencilStart[3] = {-3, -3, -3};
            const int stencilEnd[3]   = { 4,  4,  4};

            /* // use a boundary condition applied to the (current subdomain) */
            /* BoundaryCondition_CUDA<FluidBlock, FluidBlock::ElementType> bc(stencilStart, stencilEnd, src0, current_iz, current_length); */
            /* bc.template applyBC_absorbing<0,0>(buffer->xghost_l, &GPUProcessing::idx_xghosts< 3,                  FluidBlock::sizeY, 3>); */
            /* bc.template applyBC_absorbing<0,1>(buffer->xghost_r, &GPUProcessing::idx_xghosts< -FluidBlock::sizeX, FluidBlock::sizeY, 3>); */
            /* bc.template applyBC_absorbing<1,0>(buffer->yghost_l, &GPUProcessing::idx_yghosts< 3,                  FluidBlock::sizeX, 3>); */
            /* bc.template applyBC_absorbing<1,1>(buffer->yghost_r, &GPUProcessing::idx_yghosts< -FluidBlock::sizeY, FluidBlock::sizeX, 3>); */
        }


        void _compute_z_ghosts_l(const Real * const src0)
        {
            // Compute the left zghost for the current subdomain
            const int stencilStart[3] = {-3, -3, -3};
            const int stencilEnd[3]   = { 4,  4,  4};

            /* BoundaryCondition_CUDA<FluidBlock, FluidBlock::ElementType> bc(stencilStart, stencilEnd, src0); */
            /* bc.template applyBC_absorbing<2,0>(buffer->zghost_l, &GPUProcessing::idx_zghosts< 3, FluidBlock::sizeX, FluidBlock::sizeY>); */
        }


        void _compute_z_ghosts_r(const Real * const src0)
        {
            // Compute the right zghost for the current subdomain
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
            Real max_sos(const Real * const src0, const uint_t gptfloats)
            {
                /* *
                 * 1.) Init maxSOS = 0 (mapped integer -> zero-copy)
                 * 2.) Convert AOS->SOA for the FIRST/SINGLE subdomain
                 * 3.) Process all subdomains
                 * 4.) Synchronize stream to make sure reduction is complete
                 * */

                ///////////////////////////////////////////////////////////////////
                // 1.)
                ///////////////////////////////////////////////////////////////////
                *maxSOS = 0;

                ///////////////////////////////////////////////////////////////////
                // 2.)
                ///////////////////////////////////////////////////////////////////
                Timer timer;
                timer.start();
                _copy_AOS_to_SOA(buffer->SOA, src0, gptfloats, SLICE_GPU * current_length);
                const double t1 = timer.stop();
                if (chatty)
                    printf("[AOS->SOA SUBDOMAIN %d TAKES %f sec]\n", current_chunk_id, t1);

                ///////////////////////////////////////////////////////////////////
                // 3.)
                ///////////////////////////////////////////////////////////////////
                for (int i = 0; i < N_chunks; ++i)
                    _process_subdomain<Ksos>(src0, gptfloats);

                ///////////////////////////////////////////////////////////////////
                // 4.)
                ///////////////////////////////////////////////////////////////////
                this->syncStream(GPU::streamID::S1);

                union {Real f; int i;} ret;
                ret.i = *maxSOS;
                return ret.f;
            }


        template <typename Kflow, typename Kupdate>
            void process_all(const Real a, const Real b, const Real dtinvh, Real * const src0, Real * const tmp0, const uint_t gptfloats)
            {
                /* *
                 * 1.) Extract x/yghosts for current chunk and upload to GPU
                 * 2.) Copy left (and right for SINGLE) ghosts (hidden by 1.)
                 * 3.) Upload GPU input for FIRST/SINGLE subdomain (3DArrays)
                 * 4.) Process all subdomains
                 * 5.) SOA->AOS of GPU updated solution for LAST/SINGLE subdomain
                 * 6.) If subdomain is SINGLE: AOS->SOA of interior nodes (can not be hidden for SINGLE case)
                 * */

                ///////////////////////////////////////////////////////////////////
                // 1.)
                ///////////////////////////////////////////////////////////////////
                _compute_xy_ghosts(src0);
                GPU::upload_xy_ghosts(N_xyghosts,
                        buffer->xghost_l, buffer->xghost_r,
                        buffer->yghost_l, buffer->yghost_r);

                ///////////////////////////////////////////////////////////////////
                // 2.)
                ///////////////////////////////////////////////////////////////////
                // copy left ghosts always
                _compute_z_ghosts_l(src0);
                for (int i = 0; i < 7; ++i)
                    memcpy(buffer->SOA[i], buffer->zghost_l[i], N_zghosts*sizeof(Real));

                // right ghosts are conditional
                switch (chunk_state)
                {
                    case SINGLE:
                        _compute_z_ghosts_r(src0);
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
                    _process_subdomain<Kflow, Kupdate>(a, b, dtinvh, src0, tmp0, gptfloats);

                ///////////////////////////////////////////////////////////////////
                // 5.)
                ///////////////////////////////////////////////////////////////////
                // Copy back SOA->AOS of previous subdomain
                Timer timer;
                const uint_t prevOFFSET = SLICE_GPU * previous_iz * gptfloats;

                // GPU rhs into tmp (d2h finishes first for rhs)
                GPU::d2h_rhs_wait();
                timer.start();
                _copy_SOA_to_AOS(tmp0 + prevOFFSET, previous_buffer->tmpSOA, gptfloats, SLICE_GPU * previous_length);
                const double t4 = timer.stop();
                if (chatty)
                    printf("[tmpSOA->tmpAOS OF SUBDOMAIN %d TAKES %f sec]\n", previous_chunk_id, t4);

                // GPU update into src (a.k.a data)
                GPU::d2h_tmp_wait();
                timer.start();
                _copy_SOA_to_AOS(src0 + prevOFFSET, previous_buffer->SOA, gptfloats, SLICE_GPU * previous_length);
                const double t1 = timer.stop();
                if (chatty)
                    printf("[SOA->AOS OF SUBDOMAIN %d TAKES %f sec]\n", previous_chunk_id, t1);

                ///////////////////////////////////////////////////////////////////
                // 6.)
                ///////////////////////////////////////////////////////////////////
                switch (chunk_state)
                {
                    case SINGLE:
                        RealPtrVec_t first_interior(7); // pointer to first interior node
                        for (int i = 0; i < 7; ++i)
                            first_interior[i] = buffer->SOA[i] + N_zghosts;

                        timer.start();
                        _copy_AOS_to_SOA(first_interior, src0, gptfloats, SLICE_GPU * current_length);
                        const double t3 = timer.stop();
                        if (chatty)
                            printf("[AOS->SOA SUBDOMAIN %d TAKES %f sec]\n", current_chunk_id, t3);
                        break;
                }
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

/* *
 * GPU.h
 *
 * Created by Fabian Wermelinger on 6/24/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include <vector>
#include "Types.h"
#include "Profiler.h"

// flux kernels
#ifndef _NTHREADS_
#define _NTHREADS_ 64
#endif

// extra-term kernels (for x-direction only)
#ifndef _TILE_DIM_
#define _TILE_DIM_   32
#endif
#ifndef _BLOCK_ROWS_
#define _BLOCK_ROWS_ 8
#endif

#ifndef _PAGEABLE_HOST_MEM_
#include "cudaHostAllocator.h"
typedef std::vector<Real, cudaHostAllocator<Real> > cuda_vector_t;
#else
typedef std::vector<Real> cuda_vector_t;
#endif

// TODO: REMOVE
#if defined(_CUDA_TIMER_)
#define tCUDA_START(stream) { \
    GPUtimer tk; \
    tk.start(stream);
#define tCUDA_STOP(stream,msg) \
    tk.stop(stream); \
    tk.print(msg); }
#else
#define tCUDA_START(stream)
#define tCUDA_STOP(stream,msg)
#endif

namespace GPU
{
    // Named events
    enum {H2D_3DARRAY=0};

    // TODO: REMOVE
    enum streamID {S1, S2};

    extern Profiler profiler;

    ///////////////////////////////////////////////////////////////////////////
    // General GPU household -> Memory management, Streams, H2D/D2H, stats
    // Implementation: GPUhousehold.cu
    ///////////////////////////////////////////////////////////////////////////
    // alloc/dealloc
    void alloc(void** h_maxSOS, const uint_t nslices, const bool isroot = true);
    void dealloc(const bool isroot = true);

    // PCIe transfers
    void upload_xy_ghosts(const uint_t Nxghost, const RealPtrVec_t& xghost_l, const RealPtrVec_t& xghost_r,
            const uint_t Nyghost, const RealPtrVec_t& yghost_l, const RealPtrVec_t& yghost_r, const int s_id);
    void h2d_3DArray(const RealPtrVec_t& src, const uint_t nslices, const int s_id);
    void h2d_tmp(const RealPtrVec_t& src, const uint_t N);
    void d2h_rhs(RealPtrVec_t& dst, const uint_t N);
    void d2h_tmp(RealPtrVec_t& dst, const uint_t N);

    // sync
    void h2d_3DArray_wait();
    void d2h_rhs_wait();
    void d2h_tmp_wait();
    void syncGPU();
    void syncStream(streamID s);

    // stats
    void tell_memUsage_GPU();
    void tell_GPU();

    ///////////////////////////////////////////////////////////////////////////
    // GPU kernel wrappers
    // Implementation: GPUkernels.cu
    ///////////////////////////////////////////////////////////////////////////
    void bind_textures();
    void unbind_textures();
    void xflux(const uint_t nslices, const uint_t global_iz);
    void yflux(const uint_t nslices, const uint_t global_iz);
    void zflux(const uint_t nslices);
    void divergence(const Real a, const Real dtinvh, const uint_t nslices);
    void update(const Real b, const uint_t nslices);
    void MaxSpeedOfSound(const uint_t nslices);

    // Test Kernel wrapper
    void TestKernel();
}

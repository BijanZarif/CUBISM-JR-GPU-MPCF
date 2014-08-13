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

namespace GPU
{
    extern Profiler profiler;

    ///////////////////////////////////////////////////////////////////////////
    // General GPU household -> Memory management, Streams, H2D/D2H, stats
    // Implementation: GPUhousehold.cu
    ///////////////////////////////////////////////////////////////////////////
    // alloc/dealloc
    void alloc(void** h_maxSOS, const uint_t nslices, const bool isroot = true);
    void dealloc(const bool isroot = true);

    // PCIe transfers
    void upload_xy_ghosts(const uint_t Nxghost, const real_vector_t& xghost_l, const real_vector_t& xghost_r,
            const uint_t Nyghost, const real_vector_t& yghost_l, const real_vector_t& yghost_r, const int s_id);
    void h2d_3DArray(const real_vector_t& src, const uint_t nslices, const int s_id);
    void d2h_divF(real_vector_t& dst, const uint_t N, const int s_id);

    // sync
    void wait_h2d(const int e_id);
    void wait_d2h(const int e_id);
    void syncGPU();
    void syncStream(const int s_id);

    // stats
    void tell_memUsage_GPU();
    void tell_GPU();

    ///////////////////////////////////////////////////////////////////////////
    // GPU kernel wrappers
    // Implementation: GPUkernels.cu
    ///////////////////////////////////////////////////////////////////////////
    void bind_textures();
    void unbind_textures();
    void xflux(const uint_t nslices, const uint_t global_iz, const int s_id);
    void yflux(const uint_t nslices, const uint_t global_iz, const int s_id);
    void zflux(const uint_t nslices, const int s_id);
    void MaxSpeedOfSound(const uint_t nslices, const int s_id);

    // Test Kernel wrapper
    void TestKernel();
}

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

#define _WARPSIZE_ 32

// flux kernels, TODO: REMOVE
#ifndef _NTHREADS_
#define _NTHREADS_ (4*_WARPSIZE_)
#endif

// extra-term kernels (for x-direction only)
#ifndef _TILE_DIM_
#define _TILE_DIM_   _WARPSIZE_
#endif
#ifndef _BLOCK_ROWS_
#define _BLOCK_ROWS_ 4
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
    void h2d_input(
            const uint_t Nxghost, const real_vector_t& xghost_l, const real_vector_t& xghost_r,
            const uint_t Nyghost, const real_vector_t& yghost_l, const real_vector_t& yghost_r,
            const real_vector_t& src, const uint_t nslices,
            const uint_t gbuf_id, const int chunk_id);
    /* void upload_xy_ghosts(const uint_t Nxghost, const real_vector_t& xghost_l, const real_vector_t& xghost_r, */
    /*         const uint_t Nyghost, const real_vector_t& yghost_l, const real_vector_t& yghost_r, */
    /*         const uint_t gbuf_id=0, const int chunk_id=0); */
    void h2d_3DArray(const real_vector_t& src, const uint_t nslices, const uint_t gbuf_id=0, const int chunk_id=0);
    void d2h_divF(real_vector_t& dst, const uint_t N, const uint_t gbuf_t=0, const int chunk_id=0);

    // sync
    void wait_h2d(const int chunk_id);
    void wait_d2h(const int chunk_id);
    void syncGPU();
    void syncStream(const int chunk_id);

    // stats
    void tell_memUsage_GPU();
    void tell_GPU();

    ///////////////////////////////////////////////////////////////////////////
    // GPU kernel wrappers
    // Implementation: GPUkernels.cu
    ///////////////////////////////////////////////////////////////////////////
    void compute_pipe_divF(const uint_t nslices, const uint_t global_iz, const uint_t gbuf_id=0, const int chunk_id=0);
    void MaxSpeedOfSound(const uint_t nslices, const uint_t gbuf_id=0, const int chunk_id=0);

    // Test Kernel wrapper
    void TestKernel();
}

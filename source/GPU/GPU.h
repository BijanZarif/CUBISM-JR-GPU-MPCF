/* *
 * GPU.h
 *
 * Created by Fabian Wermelinger on 6/24/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include <vector>
#include "Types.h"

#ifndef _NTHREADS_
#define _NTHREADS_ 64
#endif


#ifndef _PAGEABLE_HOST_MEM_
#include "cudaHostAllocator.h"
typedef std::vector<Real, cudaHostAllocator<Real> > cuda_vector_t;
#else
typedef std::vector<Real> cuda_vector_t;
#endif


// include these declarations in host code
namespace GPU
{
    enum streamID {S1, S2};

    ///////////////////////////////////////////////////////////////////////////
    // General GPU household -> Memory management, Streams, H2D/D2H, stats
    // Implementation: GPUhousehold.cu
    ///////////////////////////////////////////////////////////////////////////
    extern "C"
    {
        // alloc/dealloc
        void alloc(void** h_maxSOS, const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t BSZ_GPU, const uint_t CHUNK_WIDTH, const bool isroot = true);
        void dealloc(const bool isroot = true);

        // PCIe transfers
        void upload_xy_ghosts(const uint_t Nxghost, const RealPtrVec_t& xghost_l, const RealPtrVec_t& xghost_r,
                const uint_t Nyghost, const RealPtrVec_t& yghost_l, const RealPtrVec_t& yghost_r);
        void h2d_3DArray(const RealPtrVec_t& src, const uint_t NX, const uint_t NY, const uint_t NZ);
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
    }

    ///////////////////////////////////////////////////////////////////////////
    // GPU kernel wrappers
    // Implementation: GPUkernels.cu
    ///////////////////////////////////////////////////////////////////////////
    extern "C"
    {
        void bind_textures();
        void unbind_textures();
        void xflux(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH, const uint_t global_iz);
        void yflux(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH, const uint_t global_iz);
        void zflux(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH);
        void divergence(const Real a, const Real dtinvh, const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH);
        void update(const Real b, const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH);
        void MaxSpeedOfSound(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH);

        // Test Kernel wrapper
        void TestKernel();
    }
}

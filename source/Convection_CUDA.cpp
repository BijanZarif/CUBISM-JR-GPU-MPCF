/*
 *  Convection_CUDA.cpp
 *  MPCFcore
 *
 *  Created by Fabian Wermelinger on 5/22/14.
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */

#include "Convection_CUDA.h"
#include "GPU.h"

Convection_CUDA::Convection_CUDA(const Real a, const Real dtinvh) : a(a), dtinvh(dtinvh) { }

void Convection_CUDA::compute(const unsigned int BSX_GPU, const unsigned int BSY_GPU, const unsigned int CHUNK_WIDTH, const unsigned int global_iz)
{
    GPU::bind_textures();
    GPU::xflux(BSX_GPU, BSY_GPU, CHUNK_WIDTH, global_iz);
    GPU::yflux(BSX_GPU, BSY_GPU, CHUNK_WIDTH, global_iz);
    GPU::zflux(BSX_GPU, BSY_GPU, CHUNK_WIDTH);
    GPU::divergence(a, dtinvh, BSX_GPU, BSY_GPU, CHUNK_WIDTH);
    GPU::unbind_textures();
}

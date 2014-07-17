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

void Convection_CUDA::compute(const uint_t nslices, const uint_t global_iz)
{
    GPU::bind_textures();
    GPU::xflux(nslices, global_iz);
    GPU::yflux(nslices, global_iz);
    GPU::zflux(nslices);
    GPU::divergence(a, dtinvh, nslices);
    GPU::unbind_textures();
}

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

void Convection_CUDA::compute(const uint_t nslices, const uint_t global_iz, const int s_id)
{
    GPU::bind_textures();
    GPU::xflux(nslices, global_iz, s_id);
    GPU::yflux(nslices, global_iz, s_id);
    GPU::zflux(nslices, s_id);
    GPU::unbind_textures();
}

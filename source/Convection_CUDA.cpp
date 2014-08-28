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

void Convection_CUDA::compute(const uint_t nslices, const uint_t global_iz, const uint_t gbuf_id, const int chunk_id)
{
    GPU::compute_pipe_divF(nslices, global_iz, gbuf_id, chunk_id);
}

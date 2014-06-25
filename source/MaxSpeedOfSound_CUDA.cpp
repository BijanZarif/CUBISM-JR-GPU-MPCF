/*
 *  MaxSpeedOfSound_CUDA.cpp
 *  MPCFcore
 *
 *  Created by Fabian Wermelinger on 06/03/14.
 *  Copyright 2014 ETH Zurich. All rights reserved.
 */

#include "MaxSpeedOfSound_CUDA.h"
#include "GPU.h"


void MaxSpeedOfSound_CUDA::compute(const unsigned int BSX_GPU, const unsigned int BSY_GPU, const unsigned int CHUNK_WIDTH)
{
    GPU::bind_textures();
    GPU::MaxSpeedOfSound(BSX_GPU, BSY_GPU, CHUNK_WIDTH);
    GPU::unbind_textures();
}

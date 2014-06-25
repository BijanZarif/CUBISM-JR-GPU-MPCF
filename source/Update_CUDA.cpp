/*
 *  Update_CUDA.cpp
 *  MPCFcore
 *
 *  Created by Fabian Wermelinger on 6/2/14.
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */
#include "Update_CUDA.h"
#include "GPU.h"

void Update_CUDA::compute(const int BSX_GPU, const int BSY_GPU, const int CHUNK_WIDTH)
{
    GPU::bind_textures();
    GPU::update(m_b, BSX_GPU, BSY_GPU, CHUNK_WIDTH);
    GPU::unbind_textures();
}

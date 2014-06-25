/*
 *  MaxSpeedOfSound_CUDA.h
 *  MPCFcore
 *
 *  Created by Fabian Wermelinger on 05/08/14.
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */
#pragma once


class MaxSpeedOfSound_CUDA
{
    public:
        void compute(const unsigned int BSX_GPU, const unsigned int BSY_GPU, const unsigned int CHUNK_WIDTH);
};

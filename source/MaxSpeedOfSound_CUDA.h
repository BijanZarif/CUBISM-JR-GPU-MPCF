/*
 *  MaxSpeedOfSound_CUDA.h
 *  MPCFcore
 *
 *  Created by Fabian Wermelinger on 05/08/14.
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "Types.h"

class MaxSpeedOfSound_CUDA
{
    public:
        void compute(const uint_t nslices);
};

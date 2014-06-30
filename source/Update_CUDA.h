/*
 *  Update_CUDA.h
 *  MPCFcore
 *
 *  Created by Fabian Wermelinger on 5/2/14.
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "Types.h"

class Update_CUDA
{
    protected:
        Real m_b;

        inline bool _is_aligned(const void * const ptr, unsigned int alignment) const
        {
            return ((size_t)ptr) % alignment == 0;
        }

    public:
        Update_CUDA(const Real b = 1) : m_b(b) { }

        void compute(const int BSX_GPU, const int BSY_GPU, const int CHUNK_WIDTH);
};

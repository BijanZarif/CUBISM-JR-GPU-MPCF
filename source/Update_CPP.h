/*
 *  Update_CPP.h
 *  MPCFcore
 *
 *  Created by Fabian Wermelinger on 8/13/14.
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "Types.h"

class Update_CPP
{
    protected:
        Real m_a, m_b, m_dtinvh;

        inline bool _is_aligned(const void * const ptr, unsigned int alignment) const
        {
            return ((size_t)ptr) % alignment == 0;
        }

    public:
        Update_CPP(const Real a, const Real b, const Real dtinvh) : m_a(a), m_b(b), m_dtinvh(dtinvh) { }

        void compute(real_vector_t& src, real_vector_t& tmp, real_vector_t& divF, const uint_t offset, const uint_t N);
};

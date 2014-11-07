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
#include <cmath>

class Update_CPP
{
protected:
    const Real m_a;
    const Real m_b;
    const Real m_dtinvh;

    // state correction
    const Real m_alpha;
    const Real m_beta;
    Real m_min_r, m_min_G, m_min_P;

    inline bool _is_aligned(const void * const ptr, unsigned int alignment) const
    {
        return ((size_t)ptr) % alignment == 0;
    }

public:
    Update_CPP(const Real a, const Real b, const Real dtinvh, const Real alpha=-3.0, const Real beta=-4.0) :
        m_a(a), m_b(b), m_dtinvh(dtinvh), m_alpha(alpha), m_beta(beta)
    {
        const Real G1 = 1.0 / (MaterialDictionary::gamma1 - 1.0);
        const Real G2 = 1.0 / (MaterialDictionary::gamma2 - 1.0);
        const Real P1 = MaterialDictionary::gamma1 * MaterialDictionary::pc1 * G1;
        const Real P2 = MaterialDictionary::gamma2 * MaterialDictionary::pc2 * G2;
        m_min_r = std::min(MaterialDictionary::rho1, MaterialDictionary::rho2);
        m_min_G = std::min(G1, G2);
        m_min_P = std::min(P1, P2);
    }

    void compute(real_vector_t& src, real_vector_t& tmp, real_vector_t& divF, const uint_t offset, const uint_t N) const;
    void state(real_vector_t& src, const uint_t offset, const uint_t N) const;
};

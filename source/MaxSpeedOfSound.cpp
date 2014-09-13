/*
 *  MaxSpeedOfSound.cpp
 *  MPCFcore
 *
 *  Created by Diego Rossinelli on 6/15/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#include <cstdlib>
#include <cmath>
#include <assert.h>
#include <algorithm>
#include <iostream>

#include "MaxSpeedOfSound.h"

Real MaxSpeedOfSound_CPP::compute(const real_vector_t& src) const
{
    const int N = _BLOCKSIZEX_ * _BLOCKSIZEY_ * _BLOCKSIZEZ_;
    Real sos = 0;

    const Real * const r_p = src[0];
    const Real * const u_p = src[1];
    const Real * const v_p = src[2];
    const Real * const w_p = src[3];
    const Real * const e_p = src[4];
    const Real * const G_p = src[5];
    const Real * const P_p = src[6];

    for(int i=0; i<N; ++i)
    {
        const Real r = r_p[i];
        const Real u = u_p[i];
        const Real v = v_p[i];
        const Real w = w_p[i];
        const Real e = e_p[i];
        const Real G = G_p[i];
        const Real P = P_p[i];

        assert(r>0);
        assert(e>0);

        assert(!isnan(r));
        assert(!isnan(u));
        assert(!isnan(v));
        assert(!isnan(w));
        assert(!isnan(e));
        assert(!isnan(G));
        assert(!isnan(P));

        const Real p = (e - (u*u + v*v + w*w)*((Real)0.5/r) - P)/G;

        const Real c = std::sqrt(((p+P)/G+p)/r);

        assert(!isnan(p));
        assert(c > 0 && !isnan(c));

        sos = std::max(sos, c + std::max(std::max(std::abs(u), std::abs(v)), std::abs(w))/r);
    }

    return sos;
}

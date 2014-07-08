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

Real MaxSpeedOfSound_CPP::compute(const RealPtrVec_t& src) const
{
    const int N = _BLOCKSIZEX_ * _BLOCKSIZEY_ * _BLOCKSIZEZ_;
    Real sos = 0;

    for(int i=0; i<N; ++i)
    {
        const Real r = src[0][i];
        const Real u = src[1][i];
        const Real v = src[2][i];
        const Real w = src[3][i];
        const Real e = src[4][i];
        const Real G = src[5][i];
        const Real P = src[6][i];

        assert(r>0);
        assert(e>0);

        assert(!isnan(r));
        assert(!isnan(u));
        assert(!isnan(v));
        assert(!isnan(w));
        assert(!isnan(e));
        assert(!isnan(G));
        assert(!isnan(P));

        const Real p = (e - (u*u + v*v + w*w)*(0.5/r) - P)/G;

        const Real c = std::sqrt(((p+P)/G+p)/r);

        assert(!isnan(p));
        assert(c > 0 && !isnan(c));

        sos = std::max(sos, c + std::max(std::max(std::abs(u), std::abs(v)), std::abs(w))/r);
    }

    return sos;
}

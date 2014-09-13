/* File        : MaxSpeedOfSound_QPX.h */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Fri 12 Sep 2014 04:22:27 PM CEST */
/* Modified    : Fri 12 Sep 2014 05:21:45 PM CEST */
/* Description : Vectorized max SOS computation */
#pragma once

#include "MaxSpeedOfSound.h"
#include "QPXEMU.h"
#include <cmath>

class MaxSpeedOfSound_QPX : public MaxSpeedOfSound_CPP
{
    inline vector4double _compute_p(const vector4double invr, const vector4double e,
            const vector4double invG, const vector4double P,
            const vector4double speed2) const
    {
        const vector4double tmp = vec_sub(e, P);
        return vec_mul(invG, vec_madd(vec_mul(vec_splats(-0.5f), invr), speed2, tmp));
    }

    inline vector4double _compute_c(const vector4double invr, const vector4double p,
            const vector4double invG, const vector4double P) const
    {
        vector4double tmp = vec_madd(invG, vec_add(p, P), p);
        return mysqrt<preclevel>(vec_mul(invr, tmp));
    }

    inline vector4double _sweep4(Real * const r, Real * const u, Real * const v, Real * const w, Real * const e, Real * const G, Real * const P) const
    {
        vector4double data0 = vec_lda(0L, r);
        vector4double data1 = vec_lda(0L, u);
        vector4double data2 = vec_lda(0L, v);
        vector4double data3 = vec_lda(0L, w);

        const vector4double invr = myreciprocal<preclevel>(data0);
        const vector4double speed2 = vec_madd(data1, data1, vec_madd(data2, data2, vec_mul(data3, data3)));
        const vector4double maxvel = mymax(vec_abs(data1), mymax(vec_abs(data2), vec_abs(data3))) ;

        data0 = vec_lda(0L, e);
        data1 = vec_lda(0L, G);
        data2 = vec_lda(0L, P);

        const vector4double invG = myreciprocal<preclevel>(data1);
        const vector4double p = _compute_p(invr, data0, invG, data2, speed2);
        const vector4double c = _compute_c(invr, p, invG, data2);

        return vec_madd(maxvel, invr, c);
    }


    public:
        Real compute(const real_vector_t& src) const
        {
            // TODO: check N%4 == 0
            enum {N = _BLOCKSIZEX_ * _BLOCKSIZEY_ * _BLOCKSIZEZ_};

            Real * const r = src[0];
            Real * const u = src[1];
            Real * const v = src[2];
            Real * const w = src[3];
            Real * const e = src[4];
            Real * const G = src[5];
            Real * const P = src[6];

            vector4double sos4 = vec_splats(0);

            for(int i=0; i < N; i += 4)
                sos4 = mymax(sos4, _sweep4(r+i, u+i, v+i, w+i, e+i, G+i, P+i));

            sos4 = mymax(sos4, vec_perm(sos4, sos4, vec_gpci(2323)));

            return std::max(vec_extract(sos4, 0), vec_extract(sos4, 1));
        }
};

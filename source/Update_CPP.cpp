/*
 *  Update_CPP.cpp
 *  MPCFcore
 *
 *  Created by Fabian Wermelinger on 8/13/14.
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */
#include "Update_CPP.h"
#include <cassert>
#include <omp.h>

void Update_CPP::compute(RealPtrVec_t& src, RealPtrVec_t& tmp, RealPtrVec_t& divF, const uint_t offset, const uint_t N)
{
    /* *
     * 1.) tmp <- a * tmp - dtinvh * divF
     * 2.) src <- b * tmp + src
     * */
    Real * const r = &src[0][offset]; Real * const u = &src[1][offset]; Real * const v = &src[2][offset];
    Real * const w = &src[3][offset]; Real * const e = &src[4][offset]; Real * const G = &src[5][offset];
    Real * const P = &src[6][offset];

    Real * const rhs_r = &tmp[0][offset]; Real * const rhs_u = &tmp[1][offset]; Real * const rhs_v = &tmp[2][offset];
    Real * const rhs_w = &tmp[3][offset]; Real * const rhs_e = &tmp[4][offset]; Real * const rhs_G = &tmp[5][offset];
    Real * const rhs_P = &tmp[6][offset];

    Real * const divFr = divF[0]; Real * const divFu = divF[1]; Real * const divFv = divF[2];
    Real * const divFw = divF[3]; Real * const divFe = divF[4]; Real * const divFG = divF[5];
    Real * const divFP = divF[6];

#pragma omp parallel for
    for (uint_t i = 0; i < N; ++i)
    {
        // 1.)
        const Real fr = -m_dtinvh*divFr[i];
        const Real fu = -m_dtinvh*divFu[i];
        const Real fv = -m_dtinvh*divFv[i];
        const Real fw = -m_dtinvh*divFw[i];
        const Real fe = -m_dtinvh*divFe[i];
        const Real fG = -m_dtinvh*divFG[i];
        const Real fP = -m_dtinvh*divFP[i];

        Real rr = rhs_r[i];
        Real ru = rhs_u[i];
        Real rv = rhs_v[i];
        Real rw = rhs_w[i];
        Real re = rhs_e[i];
        Real rG = rhs_G[i];
        Real rP = rhs_P[i];
        rr = m_a * rr + fr;
        ru = m_a * ru + fu;
        rv = m_a * rv + fv;
        rw = m_a * rw + fw;
        re = m_a * re + fe;
        rG = m_a * rG + fG;
        rP = m_a * rP + fP;

        // 2.)
        r[i] += m_b * rr;
        u[i] += m_b * ru;
        v[i] += m_b * rv;
        w[i] += m_b * rw;
        e[i] += m_b * re;
        G[i] += m_b * rG;
        P[i] += m_b * rP;

        rhs_r[i] = rr;
        rhs_u[i] = ru;
        rhs_v[i] = rv;
        rhs_w[i] = rw;
        rhs_e[i] = re;
        rhs_G[i] = rG;
        rhs_P[i] = rP;
    }
}

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
#include <cmath>
using std::max;
using std::min;
using std::abs;

void Update_CPP::compute(real_vector_t& src, real_vector_t& tmp, real_vector_t& divF, const uint_t offset, const uint_t N)
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

        Real r_new = rhs_r[i];
        Real u_new = rhs_u[i];
        Real v_new = rhs_v[i];
        Real w_new = rhs_w[i];
        Real e_new = rhs_e[i];
        Real G_new = rhs_G[i];
        Real P_new = rhs_P[i];
        r_new = m_a * r_new + fr;
        u_new = m_a * u_new + fu;
        v_new = m_a * v_new + fv;
        w_new = m_a * w_new + fw;
        e_new = m_a * e_new + fe;
        G_new = m_a * G_new + fG;
        P_new = m_a * P_new + fP;

        rhs_r[i] = r_new;
        rhs_u[i] = u_new;
        rhs_v[i] = v_new;
        rhs_w[i] = w_new;
        rhs_e[i] = e_new;
        rhs_G[i] = G_new;
        rhs_P[i] = P_new;

        // 2.)
#ifndef _STATE_
        r[i] += m_b * r_new;
        u[i] += m_b * u_new;
        v[i] += m_b * v_new;
        w[i] += m_b * w_new;
        e[i] += m_b * e_new;
        G[i] += m_b * G_new;
        P[i] += m_b * P_new;
#else
        r_new = m_b * r_new + r[i];
        u_new = m_b * u_new + u[i];
        v_new = m_b * v_new + v[i];
        w_new = m_b * w_new + w[i];
        e_new = m_b * e_new + e[i];
        G_new = m_b * G_new + G[i];
        P_new = m_b * P_new + P[i];

        // update state based on cubism Update_State kernel
        r[i] = max(r_new, static_cast<Real>(1.0));    // change rho
        G[i] = max(G_new, static_cast<Real>(m_min_G));// change G
        P[i] = max(P_new, static_cast<Real>(m_min_P));// change P
        u[i] = u_new;
        v[i] = v_new;
        w[i] = w_new;

        const Real ke = 0.5*(u_new*u_new + v_new*v_new + w_new*w_new)/r_new; // whatever ke we had before
        const Real pressure = (e_new - P_new - ke)/G_new; // whatever pressure we had before

        /* if (P[i]/(static_cast<Real>(1.0) + G[i]) < -2*pressure) // if it was still bad with new P and new G */
        /* { */
        /*     /1* const Real difference = -16.86 * pressure * (static_cast<Real>(1.0) + G[i]) - P[i]; *1/ */
        /*     /1* P[i] += abs(difference); // change P again *1/ */
        /*     G[i] *= static_cast<Real>(0.2); */
        /* } */
        if (G[i] > 2.55) // if it was still bad with new P and new G
        {
            /* const Real diff_G = 0.7*G[i] - 2.55; */
            const Real diff_P = -4 * pressure * (static_cast<Real>(1.0) + G[i]) - P[i];
            G[i] = 2.5
            P[i] += abs(diff_P); // change P again
        }

        e[i] = pressure * G[i] + P[i] + ke; // update e
#endif
    }
}

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

// TODO: remove this
#include <cstdio>
static FILE *correctionFile;
static uint_t been_here = 0;


void Update_CPP::compute(real_vector_t& src, real_vector_t& tmp, real_vector_t& divF, const uint_t offset, const uint_t N)
{
    /* *
     * 1.) tmp <- a * tmp - dtinvh * divF
     * 2.) src <- b * tmp + src
     * */
    assert(src.size() == tmp.size() && tmp.size() == divF.size());

    for (int c=0; c < (int)src.size(); ++c)
    {
        Real* const U      = &src[c][offset];
        Real* const rhs    = &tmp[c][offset];
        Real* const divF_U = divF[c];

#pragma omp parallel for
        for (uint_t i = 0; i < N; ++i)
        {
            Real U_new = rhs[i];
            const Real U_old = U[i];
            const Real rhs_new = -m_dtinvh * divF_U[i];

            // 1.)
            U_new = m_a * U_new + rhs_new;
            rhs[i] = U_new;

            // 2.)
            U[i] = U_old + m_b * U_new;
        }
    }
}


void Update_CPP::state(real_vector_t& src, const uint_t offset, const uint_t N)
{
    Real* const r = &src[0][offset];
    Real* const u = &src[1][offset];
    Real* const v = &src[2][offset];
    Real* const w = &src[3][offset];
    Real* const e = &src[4][offset];
    Real* const G = &src[5][offset];
    Real* const P = &src[6][offset];

    correctionFile = fopen("corrections.dat", "a");

#pragma omp parallel for
        for (uint_t i = 0; i < N; ++i)
        {
            const Real r_new = r[i];
            const Real u_new = u[i];
            const Real v_new = v[i];
            const Real w_new = w[i];
            const Real e_new = e[i];
            const Real G_new = G[i];
            const Real P_new = P[i];

            // change material state
            r[i] = max(r_new, static_cast<Real>(m_min_r));// change rho
            G[i] = max(G_new, static_cast<Real>(m_min_G));// change G
            P[i] = max(P_new, static_cast<Real>(m_min_P));// change P

            // current ke and pressure
            const Real ke = static_cast<Real>(0.5)*(u_new*u_new + v_new*v_new + w_new*w_new)/r_new; // whatever ke we had before
            const Real pressure = (e_new - P_new - ke)/G_new; // whatever pressure we had before

            // apply correction to current material state, based on the thought
            // that energy > 0 for all t
            if (P[i]/(static_cast<Real>(1.0) + G[i]) < static_cast<Real>(-2.0)*pressure) // if it was still bad with new P and new G
            {
#pragma omp atomic
                ++been_here;
#pragma omp critical
                fprintf(correctionFile, "%d\t%e\t%e\t", been_here, G[i]*pressure, P[i]);

                const Real difference = static_cast<Real>(-4.0) * pressure * (static_cast<Real>(1.0) + G[i]) - P[i];
                P[i] += difference; // change P again

#pragma omp critical
                fprintf(correctionFile, "%e\n", P[i]);
            }

            // (possibly corrected) energy
            e[i] = pressure * G[i] + P[i] + ke; // update e
        }

        fclose(correctionFile);
}


        /* r_new = m_b * r_new + r[i]; */
        /* u_new = m_b * u_new + u[i]; */
        /* v_new = m_b * v_new + v[i]; */
        /* w_new = m_b * w_new + w[i]; */
        /* e_new = m_b * e_new + e[i]; */
        /* G_new = m_b * G_new + G[i]; */
        /* P_new = m_b * P_new + P[i]; */

        /* // update state based on cubism Update_State kernel */
        /* r[i] = max(r_new, static_cast<Real>(m_min_r));// change rho */
        /* G[i] = max(G_new, static_cast<Real>(m_min_G));// change G */
        /* P[i] = max(P_new, static_cast<Real>(m_min_P));// change P */
        /* u[i] = u_new; */
        /* v[i] = v_new; */
        /* w[i] = w_new; */

        /* const Real ke = static_cast<Real>(0.5)*(u_new*u_new + v_new*v_new + w_new*w_new)/r_new; // whatever ke we had before */
        /* const Real pressure = (e_new - P_new - ke)/G_new; // whatever pressure we had before */

        /* /1* if (P[i]/(static_cast<Real>(1.0) + G[i]) < static_cast<Real>(-2.0)*pressure) // if it was still bad with new P and new G *1/ */
        /* /1* { *1/ */
        /* /1*     /2* const Real difference = static_cast<Real>(-8.0) * pressure * (static_cast<Real>(1.0) + G[i]) - P[i]; *2/ *1/ */
        /* /1*     const Real difference = static_cast<Real>(-8.0) * pressure * (static_cast<Real>(1.0) + G[i]) - P[i]; *1/ */
        /* /1*     /2* P[i] += abs(difference); // change P again *2/ *1/ */
        /* /1*     P[i] += difference; // change P again *1/ */
        /* /1* } *1/ */

        /* /1* if (P[i] < static_cast<Real>(-2.5)*G[i]*pressure) // if it was still bad with new P and new G *1/ */
        /* if (P[i]/(static_cast<Real>(1.0) + G[i]) < static_cast<Real>(-2.0)*pressure) // if it was still bad with new P and new G */
        /* { */
/* #pragma omp atomic */
        /*     ++been_here; */
/* #pragma omp critical */
        /*     fprintf(correctionFile, "%d\t%e\t%e\t", been_here, G[i]*pressure, P[i]); */

        /*     /1* const Real difference = static_cast<Real>(-8.0)*G[i]*pressure - P[i]; *1/ */

        /*     // factor of 2 increase? from 256 nodes to 512, does it scale? */
        /*     // difference is purely based on EOS */
        /*     /1* const Real difference = static_cast<Real>(-1.2)*G[i]*pressure - P[i]; *1/ */


        /*     // Babak's difference */
        /*     const Real difference = static_cast<Real>(-4.0) * pressure * (static_cast<Real>(1.0) + G[i]) - P[i]; */


        /*     P[i] += difference; // change P again */


/* #pragma omp critical */
        /*     fprintf(correctionFile, "%e\n", P[i]); */
        /* } */


        /* /1* const Real mygamma = (G[i] + static_cast<Real>(1.0)) / G[i]; *1/ */
        /* /1* if (mygamma < static_cast<Real>(1.38)) // if it was still bad with new P and new G *1/ */
        /* /1* { *1/ */
        /* /1*     const Real difference = -static_cast<Real>(4.0) * pressure * (static_cast<Real>(1.0) + G[i]) - P[i]; *1/ */
        /* /1*     P[i] += abs(difference); // change P again *1/ */
        /* /1* } *1/ */


        /* /1* if (G[i] > 2.55) // if it was still bad with new P and new G *1/ */
        /* /1* { *1/ */
        /* /1*     /2* const Real diff_G = 0.7*G[i] - 2.55; *2/ *1/ */
        /* /1*     const Real diff_P = -4 * pressure * (static_cast<Real>(1.0) + G[i]) - P[i]; *1/ */
        /* /1*     G[i] = 2.5 *1/ */
        /* /1*     P[i] += abs(diff_P); // change P again *1/ */
        /* /1* } *1/ */

        /* e[i] = pressure * G[i] + P[i] + ke; // update e */
/* #endif */
    /* } */

    /* fclose(correctionFile); */

/* } */


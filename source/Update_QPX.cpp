/* File        : Update_QPX.cpp */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Sat 13 Sep 2014 09:52:43 AM CEST */
/* Modified    : Sun 28 Sep 2014 02:52:15 PM CEST */
/* Description : Copyright © 2014 ETH Zurich. All Rights Reserved. */

#include "Update_QPX.h"
#include "QPXEMU.h"
#include <cassert>

void Update_QPX::compute(real_vector_t& src, real_vector_t& tmp, real_vector_t& divF, const uint_t offset, const uint_t N)
{
    /* *
     * 1.) tmp <- a * tmp - dtinvh * divF
     * 2.) src <- b * tmp + src
     * */
    assert(src.size() == tmp.size() && tmp.size() == divF.size());

    /* const vector4double a = vec_splats(m_a); */
    /* const vector4double b = vec_splats(m_b); */
    /* const vector4double dtinvh = vec_splats(-m_dtinvh); */

    for (int c=0; c < (int)src.size(); ++c)
    {
        Real* const U      = &src[c][offset];
        Real* const rhs    = &tmp[c][offset];
        Real* const divF_U = divF[c];

        // TODO: N%4 == 0 ?
#pragma omp parallel for
        for(int i=0; i < N; i += 4)
        {
            vector4double U_new = vec_lda(0L, rhs + i);
            const vector4double U_old = vec_lda(0L, U + i);
            const vector4double rhs_new = vec_mul(vec_splats(-m_dtinvh), vec_lda(0L, divF_U + i));
            U_new = vec_madd(vec_splats(m_a), U_new, rhs_new);

            // 1.)
            vec_sta(U_new, 0L, rhs + i);

            // 2.)
            vec_sta(vec_madd(vec_splats(m_b), U_new, U_old), 0L, U + i);
        }
    }
}

void Update_QPX::state(real_vector_t& src, const uint_t offset, const uint_t N)
{
    Real* const r = &src[0][offset];
    Real* const u = &src[1][offset];
    Real* const v = &src[2][offset];
    Real* const w = &src[3][offset];
    Real* const e = &src[4][offset];
    Real* const G = &src[5][offset];
    Real* const P = &src[6][offset];

    const Real alpha = -2.0;
    const Real beta  = -4.0;

#pragma omp parallel for
    for (uint_t i = 0; i < N; i += 4)
    {
        vector4double r_old = vec_lda(0L, r+i);
        vector4double u_old = vec_lda(0L, u+i);
        vector4double v_old = vec_lda(0L, v+i);
        vector4double w_old = vec_lda(0L, w+i);
        vector4double e_old = vec_lda(0L, e+i);
        vector4double G_old = vec_lda(0L, G+i);
        vector4double P_old = vec_lda(0L, P+i);

        vector4double r_new = vec_max(r_old, vec_splats(m_min_r));
        vector4double G_new = vec_max(G_old, vec_splats(m_min_G));
        vector4double P_new = vec_max(P_old, vec_splats(m_min_P));

        const vector4double ke = vec_mul(vec_splats((Real)0.5),
                vec_mul(vec_add(vec_mul(u_old,u_old), vec_add(vec_mul(v_old,v_old), vec_mul(v_old,v_old))),
                    myreciprocal<preclevel>(r_old)));

        const vector4double pressure = vec_mul(vec_sub(vec_sub(e_old,P_old),ke), myreciprocal<preclevel>(G_old));

        const vector4double flag = vec_cmpgt(vec_mul(vec_splats(alpha),pressure), vec_mul(P_new, myreciprocal<preclevel>(vec_add(vec_splats((Real)1.0),G_new))));

        const vector4double difference = vec_msub(vec_mul(vec_splats(beta), pressure), vec_add(vec_splats((Real)1.0),G_new), P_new);

        P_new = vec_add(P_new, vec_sel(vec_splats((Real)0.0), difference, flag));

        vector4double e_new = vec_add(vec_madd(pressure,G_new, P_new), ke);

        vec_sta(r_new, 0L, r+i);
        vec_sta(G_new, 0L, G+i);
        vec_sta(P_new, 0L, P+i);
        vec_sta(e_new, 0L, e+i);
    }
}

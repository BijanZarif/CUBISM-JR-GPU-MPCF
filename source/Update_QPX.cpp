/* File        : Update_QPX.cpp */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Sat 13 Sep 2014 09:52:43 AM CEST */
/* Modified    : Sat 13 Sep 2014 10:45:34 AM CEST */
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
    Real * const r = &src[0][offset];
    Real * const u = &src[1][offset];
    Real * const v = &src[2][offset];
    Real * const w = &src[3][offset];
    Real * const e = &src[4][offset];
    Real * const G = &src[5][offset];
    Real * const P = &src[6][offset];

    Real * const rhs_r = &tmp[0][offset];
    Real * const rhs_u = &tmp[1][offset];
    Real * const rhs_v = &tmp[2][offset];
    Real * const rhs_w = &tmp[3][offset];
    Real * const rhs_e = &tmp[4][offset];
    Real * const rhs_G = &tmp[5][offset];
    Real * const rhs_P = &tmp[6][offset];

    // TODO: check that divF is 16-byte aligned
    Real * const divFr = divF[0];
    Real * const divFu = divF[1];
    Real * const divFv = divF[2];
    Real * const divFw = divF[3];
    Real * const divFe = divF[4];
    Real * const divFG = divF[5];
    Real * const divFP = divF[6];

    const vector4double a = vec_splats(m_a);
    const vector4double b = vec_splats(m_b);
    const vector4double dtinvh = vec_splats(-m_dtinvh);

    // TODO: N%4 == 0 ?
    for(int i=0; i < N; i += 4)
    {
        // batch 1
        vector4double data0 = vec_lda(0L, divFr + i);
        vector4double data1 = vec_lda(0L, divFu + i);
        vector4double data2 = vec_lda(0L, divFv + i);
        vector4double data3 = vec_lda(0L, divFw + i);

        vector4double data4 = vec_lda(0L, rhs_r + i);
        vector4double data5 = vec_lda(0L, rhs_u + i);
        vector4double data6 = vec_lda(0L, rhs_v + i);
        vector4double data7 = vec_lda(0L, rhs_w + i);

        data4 = vec_madd(a, data4, vec_mul(dtinvh, data0));
        data5 = vec_madd(a, data5, vec_mul(dtinvh, data1));
        data6 = vec_madd(a, data6, vec_mul(dtinvh, data2));
        data7 = vec_madd(a, data7, vec_mul(dtinvh, data3));

        // new RHS
        vec_sta(data4, 0L, rhs_r + i);
        vec_sta(data5, 0L, rhs_u + i);
        vec_sta(data6, 0L, rhs_v + i);
        vec_sta(data7, 0L, rhs_w + i);

        // update solution
        vec_sta(vec_madd(b, data4, vec_lda(0L, r + i)), 0L, r + i);
        vec_sta(vec_madd(b, data5, vec_lda(0L, u + i)), 0L, u + i);
        vec_sta(vec_madd(b, data6, vec_lda(0L, v + i)), 0L, v + i);
        vec_sta(vec_madd(b, data7, vec_lda(0L, w + i)), 0L, w + i);

        // batch 2
        data0 = vec_lda(0L, divFe + i);
        data1 = vec_lda(0L, divFG + i);
        data2 = vec_lda(0L, divFP + i);

        data4 = vec_lda(0L, rhs_e + i);
        data5 = vec_lda(0L, rhs_G + i);
        data6 = vec_lda(0L, rhs_P + i);

        data4 = vec_madd(a, data4, vec_mul(dtinvh, data0));
        data5 = vec_madd(a, data5, vec_mul(dtinvh, data1));
        data6 = vec_madd(a, data6, vec_mul(dtinvh, data2));

        // new RHS
        vec_sta(data4, 0L, rhs_e + i);
        vec_sta(data5, 0L, rhs_G + i);
        vec_sta(data6, 0L, rhs_P + i);

        // update solution
        vec_sta(vec_madd(b, data4, vec_lda(0L, e + i)), 0L, e + i);
        vec_sta(vec_madd(b, data5, vec_lda(0L, G + i)), 0L, G + i);
        vec_sta(vec_madd(b, data6, vec_lda(0L, P + i)), 0L, P + i);
    }
}

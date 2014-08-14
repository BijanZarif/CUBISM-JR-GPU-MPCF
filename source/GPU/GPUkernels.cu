/* *
 * GPUkernels.cu
 *
 * Created by Fabian Wermelinger on 6/25/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <assert.h>
#include <stdio.h>

#include "GPU.cuh"

#if _BLOCKSIZEX_ < 5
#error Minimum _BLOCKSIZEX_ is 5
#elif _BLOCKSIZEY_ < 5
#error Minimum _BLOCKSIZEY_ is 5
#elif _BLOCKSIZEZ_ < 1
#error Minimum _BLOCKSIZEZ_ is 1
#endif

#if NX % _TILE_DIM_ != 0
#error _BLOCKSIZEX_ should be an integer multiple of _TILE_DIM_
#endif
#if NY % _TILE_DIM_ != 0
#error _BLOCKSIZEY_ should be an integer multiple of _TILE_DIM_
#endif


///////////////////////////////////////////////////////////////////////////////
//                           GLOBAL VARIABLES                                //
///////////////////////////////////////////////////////////////////////////////
// helper storage
extern real_vector_t d_flux;
extern Real *d_Gm, *d_Gp;
extern Real *d_Pm, *d_Pp;
extern Real *d_hllc_vel;
extern Real *d_sumG, *d_sumP, *d_divU;

// max SOS
extern int *d_maxSOS;

// GPU input/output
extern struct GPU_COMM gpu_comm[_NUM_GPU_BUF_];

// use non-null stream (async)
extern cudaStream_t *stream;

// compute events
extern cudaEvent_t *event_compute;

// texture references
#include "Texture.cu"

///////////////////////////////////////////////////////////////////////////////
//                             DEVICE FUNCTIONS                              //
///////////////////////////////////////////////////////////////////////////////
__device__
inline Real _weno_pluss(const Real b, const Real c, const Real d, const Real e, const Real f)
{
    const Real wenoeps_f = (Real)WENOEPS;
#ifndef _WENO3_
    // (90 MUL/ADD/SUB + 6 DIV) = 96 FLOP
    const Real inv6 = 1.0f/6.0f;
    const Real inv3 = 1.0f/3.0f;
    const Real q1 =  10.0f*inv3;
    const Real q2 =  31.0f*inv3;
    const Real q3 =  11.0f*inv3;
    const Real q4 =  25.0f*inv3;
    const Real q5 =  19.0f*inv3;
    const Real q6 =   4.0f*inv3;
    const Real q7 =  13.0f*inv3;
    const Real q8 =   5.0f*inv3;

    const Real sum0 =  inv3*f - 7.0f*inv6*e + 11.0f*inv6*d;
    const Real sum1 = -inv6*e + 5.0f*inv6*d + inv3*c;
    const Real sum2 =  inv3*d + 5.0f*inv6*c - inv6*b;

    const Real is0 = d*(d*q1 - e*q2 + f*q3) + e*(e*q4 - f*q5) + f*f*q6;
    const Real is1 = c*(c*q6 - d*q7 + e*q8) + d*(d*q7 - e*q7) + e*e*q6;
    const Real is2 = b*(b*q6 - c*q5 + d*q3) + c*(c*q4 - d*q2) + d*d*q1;

    const Real is0plus = is0 + wenoeps_f;
    const Real is1plus = is1 + wenoeps_f;
    const Real is2plus = is2 + wenoeps_f;

    const Real alpha0 = 1.0f / (10.0f*is0plus*is0plus);
    const Real alpha1 = 6.0f * (1.0f / (10.0f*is1plus*is1plus));
    const Real alpha2 = 3.0f * (1.0f / (10.0f*is2plus*is2plus));
    const Real alphasumInv = 1.0f / (alpha0+alpha1+alpha2);

    const Real omega0 = alpha0 * alphasumInv;
    const Real omega1 = alpha1 * alphasumInv;
    const Real omega2 = 1.0f - omega0 - omega1;

    return omega0*sum0 + omega1*sum1 + omega2*sum2;

#else
    // 28 FLOP
    const Real sum0 = 1.5f*d - 0.5f*e;
    const Real sum1 = 0.5f*(d + c);

    const Real is0 = (d-e)*(d-e);
    const Real is1 = (d-c)*(d-c);

    const Real alpha0 = 1.0f / (3.0f * (is0+wenoeps_f)*(is0+wenoeps_f));
    const Real alpha1 = 2.0f * (1.0f / (3.0f * (is1+wenoeps_f)*(is1+wenoeps_f)));

    const Real omega0 = alpha0 / (alpha0+alpha1);
    const Real omega1 = 1.0f - omega0;

    return omega0*sum0 + omega1*sum1;

#endif
}


__device__
inline Real _weno_minus(const Real a, const Real b, const Real c, const Real d, const Real e)
{
    const Real wenoeps_f = (Real)WENOEPS;
#ifndef _WENO3_
    // (90 MUL/ADD/SUB + 6 DIV) = 96 FLOP
    const Real inv6 = 1.0f/6.0f;
    const Real inv3 = 1.0f/3.0f;
    const Real q1 =   4.0f*inv3;
    const Real q2 =  19.0f*inv3;
    const Real q3 =  11.0f*inv3;
    const Real q4 =  25.0f*inv3;
    const Real q5 =  31.0f*inv3;
    const Real q6 =  10.0f*inv3;
    const Real q7 =  13.0f*inv3;
    const Real q8 =   5.0f*inv3;

    const Real sum0 =  inv3*a - 7.0f*inv6*b + 11.0f*inv6*c;
    const Real sum1 = -inv6*b + 5.0f*inv6*c + inv3*d;
    const Real sum2 =  inv3*c + 5.0f*inv6*d - inv6*e;

    const Real is0 = a*(a*q1 - b*q2 + c*q3) + b*(b*q4 - c*q5) + c*c*q6;
    const Real is1 = b*(b*q1 - c*q7 + d*q8) + c*(c*q7 - d*q7) + d*d*q1;
    const Real is2 = c*(c*q6 - d*q5 + e*q3) + d*(d*q4 - e*q2) + e*e*q1;

    const Real is0plus = is0 + wenoeps_f;
    const Real is1plus = is1 + wenoeps_f;
    const Real is2plus = is2 + wenoeps_f;

    const Real alpha0 = 1.0f / (10.0f*is0plus*is0plus);
    const Real alpha1 = 6.0f * (1.0f / (10.0f*is1plus*is1plus));
    const Real alpha2 = 3.0f * (1.0f / (10.0f*is2plus*is2plus));
    const Real alphasumInv = 1.0f / (alpha0+alpha1+alpha2);

    const Real omega0 = alpha0 * alphasumInv;
    const Real omega1 = alpha1 * alphasumInv;
    const Real omega2 = 1.0f - omega0 - omega1;

    return omega0*sum0 + omega1*sum1 + omega2*sum2;

#else
    // 28 FLOP
    const Real sum0 = 1.5f*c - 0.5f*b;
    const Real sum1 = 0.5f*(c + d);

    const Real is0 = (c-b)*(c-b);
    const Real is1 = (d-c)*(d-c);

    const Real alpha0 = 1.0f / (3.0f * (is0+wenoeps_f)*(is0+wenoeps_f));
    const Real alpha1 = 2.0f * (1.0f / (3.0f * (is1+wenoeps_f)*(is1+wenoeps_f)));

    const Real omega0 = alpha0 / (alpha0+alpha1);
    const Real omega1 = 1.0f - omega0;

    return omega0*sum0 + omega1*sum1;

#endif
}


__device__
inline Real _weno_pluss_clipped(const Real b, const Real c, const Real d, const Real e, const Real f)
{
    const Real retval = _weno_pluss(b,c,d,e,f);
    const Real min_in = fminf( fminf(c,d), e );
    const Real max_in = fmaxf( fmaxf(c,d), e );
    return fminf(fmaxf(retval, min_in), max_in);
}


__device__
inline Real _weno_minus_clipped(const Real a, const Real b, const Real c, const Real d, const Real e)
{
    const Real retval = _weno_minus(a,b,c,d,e);
    const Real min_in = fminf( fminf(b,c), d );
    const Real max_in = fmaxf( fmaxf(b,c), d );
    return fminf(fmaxf(retval, min_in), max_in);
}


__device__
inline void _char_vel_einfeldt(const Real rm, const Real rp,
        const Real vm, const Real vp,
        const Real pm, const Real pp,
        const Real Gm, const Real Gp,
        const Real Pm, const Real Pp,
        Real& outm, Real& outp) // (23 MUL/ADD/SUB + 6 DIV) = 29 FLOP
{
    /* *
     * Compute upper and lower bounds of signal velocities for the Riemann
     * problem according to Einfeldt:
     *
     * 1.) Compute Rr needed for Roe averages
     * 2.) Compute speed of sound in left and right state
     * 3.) Compute speed of sound according to Einfeldt and Rr
     * 4.) Compute upper and lower signal velocities
     * */

    // 1.)
    assert(rm > 0.0f);
    assert(rp > 0.0f);
    const Real Rr   = sqrtf(rp / rm);
    const Real Rinv = 1.0f / (1.0f + Rr);

    // 2.)
    const Real cm2 = ((pm + Pm)/Gm + pm) / rm;
    const Real cp2 = ((pp + Pp)/Gp + pp) / rp;
    const Real cm  = sqrtf(cm2);
    const Real cp  = sqrtf(cp2);
    assert(!isnan(cm));
    assert(!isnan(cp));

    // 3.)
    const Real um    = vm;
    const Real up    = vp;
    const Real eta_2 = 0.5f*Rr*Rinv*Rinv;
    const Real d2    = (cm2 + Rr*cp2)*Rinv + eta_2*(up - um)*(up - um);
    const Real d     = sqrtf(d2);
    const Real u     = (um + Rr*up)*Rinv;
    assert(!isnan(d));
    assert(!isnan(u));

    // 4.)
    outm = fminf(u - d, um - cm);
    outp = fmaxf(u + d, up + cp);
}


/* *
 * Compute characteristic velocity, s^star, of the intermediate wave.  The
 * computation is based on the condition of uniform constant pressure in
 * the star region.  See P. Batten et. al., "On the choice of wavespeeds
 * for the HLLC Riemann solver", SIAM J. Sci. Comput. 18 (1997) 1553--1570
 * It is assumed s^minus and s^plus are known.
 * */
__device__
inline Real _char_vel_star(const Real rm, const Real rp,
        const Real vm, const Real vp,
        const Real pm, const Real pp,
        const Real sm, const Real sp) // (10 MUL/ADD/SUB + 1 DIV) = 11 FLOP
{
    const Real facm = rm * (sm - vm);
    const Real facp = rp * (sp - vp);
    return (pp - pm + vm*facm - vp*facp) / (facm - facp);
    /* return (pp + vm*facm - (pm + vp*facp)) / (facm - facp); */
}


__device__
inline Real _hllc_rho(const Real rm, const Real rp,
        const Real vm, const Real vp,
        const Real sm, const Real sp, const Real ss) // (21 MUL/ADD/SUB + 2 DIV) = 23 FLOP
{
    /* *
     * The flux computation is split into 4 parts:
     * 1.) Compute signum of s^*, compute s^- and s^+
     * 2.) Compute chi^* and delta of q^* and q
     * 3.) Compute trivial flux
     * 4.) Compute HLLC flux
     * */

    // 1.)
    const Real sign_star = (ss == 0.0f) ? 0.0f : ((ss < 0.0f) ? -1.0f : 1.0f);
    const Real s_minus = fminf(0.0f, sm);
    const Real s_pluss = fmaxf(0.0f, sp);

    // 2.)
    const Real chi_starm = (sm - vm) / (sm - ss);
    const Real chi_starp = (sp - vp) / (sp - ss);
    const Real qm        = rm;
    const Real qp        = rp;
    const Real q_deltam  = qm*chi_starm - qm;
    const Real q_deltap  = qp*chi_starp - qp;

    // 3.)
    const Real fm = qm*vm;
    const Real fp = qp*vp;

    // 4.)
    const Real flux = (0.5f*(1.0f + sign_star)) * (fm + s_minus*q_deltam) + (0.5f*(1.0f - sign_star)) * (fp + s_pluss*q_deltap);
    assert(!isnan(flux));
    return flux;
}


__device__
inline Real _hllc_vel(const Real rm,  const Real rp,
        const Real vm,  const Real vp,
        const Real vdm, const Real vdp,
        const Real sm,  const Real sp,  const Real ss) // (23 MUL/ADD/SUB + 2 DIV) = 25 FLOP
{
    /* *
     * The flux computation is split into 4 parts:
     * 1.) Compute signum of s^*, compute s^- and s^+
     * 2.) Compute chi^* and delta of q^* and q
     * 3.) Compute trivial flux
     * 4.) Compute HLLC flux
     * */

    // 1.)
    const Real sign_star = (ss == 0.0f) ? 0.0f : ((ss < 0.0f) ? -1.0f : 1.0f);
    const Real s_minus  = fminf(0.0f, sm);
    const Real s_pluss  = fmaxf(0.0f, sp);

    // 2.)
    const Real chi_starm = (sm - vdm) / (sm - ss);
    const Real chi_starp = (sp - vdp) / (sp - ss);
    const Real qm        = rm*vm;
    const Real qp        = rp*vp;
    const Real q_deltam  = qm*chi_starm - qm;
    const Real q_deltap  = qp*chi_starp - qp;

    // 3.)
    const Real fm = qm*vdm;
    const Real fp = qp*vdp;

    // 4.)
    const Real flux = (0.5f*(1.0f + sign_star)) * (fm + s_minus*q_deltam) + (0.5f*(1.0f - sign_star)) * (fp + s_pluss*q_deltap);
    assert(!isnan(flux));
    assert(!isnan(ss));
    assert(!isnan(sm));
    assert(!isnan(sp));
    return flux;
}


__device__
inline Real _hllc_pvel(const Real rm, const Real rp,
        const Real vm, const Real vp,
        const Real pm, const Real pp,
        const Real sm, const Real sp, const Real ss) // (27 MUL/ADD/SUB + 2 DIV) = 29 FLOP
{
    /* *
     * The flux computation is split into 4 parts:
     * 1.) Compute signum of s^*, compute s^- and s^+
     * 2.) Compute chi^* and delta of q^* and q
     * 3.) Compute trivial flux
     * 4.) Compute HLLC flux
     * */

    // 1.)
    const Real sign_star = (ss == 0.0f) ? 0.0f : ((ss < 0.0f) ? -1.0f : 1.0f);
    const Real s_minus  = fminf(0.0f, sm);
    const Real s_pluss  = fmaxf(0.0f, sp);

    // 2.)
    const Real chi_starm = (sm - vm) / (sm - ss);
    const Real chi_starp = (sp - vp) / (sp - ss);
    const Real qm        = rm*vm;
    const Real qp        = rp*vp;
    const Real q_deltam  = rm*ss*chi_starm - qm;
    const Real q_deltap  = rp*ss*chi_starp - qp;

    // 3.)
    const Real fm = qm*vm + pm;
    const Real fp = qp*vp + pp;

    // 4.)
    const Real flux = (0.5f*(1.0f + sign_star)) * (fm + s_minus*q_deltam) + (0.5f*(1.0f - sign_star)) * (fp + s_pluss*q_deltap);
    assert(!isnan(flux));
    assert(rm > 0);
    assert(rp > 0);
    return flux;
}


__device__
inline Real _hllc_e(const Real rm,  const Real rp,
        const Real vdm, const Real vdp,
        const Real v1m, const Real v1p,
        const Real v2m, const Real v2p,
        const Real pm,  const Real pp,
        const Real Gm,  const Real Gp,
        const Real Pm,  const Real Pp,
        const Real sm,  const Real sp,  const Real ss) // (55 MUL/ADD/SUB + 4 DIV) = 59 FLOP
{
    /* *
     * The flux computation is split into 4 parts:
     * 1.) Compute signum of s^*, compute s^- and s^+
     * 2.) Compute chi^* and delta of q^* and q
     * 3.) Compute trivial flux
     * 4.) Compute HLLC flux
     * */

    // 1.)
    const Real sign_star = (ss == 0.0f) ? 0.0f : ((ss < 0.0f) ? -1.0f : 1.0f);
    const Real s_minus  = fminf(0.0f, sm);
    const Real s_pluss  = fmaxf(0.0f, sp);

    // 2.)
    const Real chi_starm = (sm - vdm) / (sm - ss);
    const Real chi_starp = (sp - vdp) / (sp - ss);
    const Real qm        = Gm*pm + Pm + 0.5f*rm*(vdm*vdm + v1m*v1m + v2m*v2m);
    const Real qp        = Gp*pp + Pp + 0.5f*rp*(vdp*vdp + v1p*v1p + v2p*v2p);
    const Real q_deltam  = chi_starm*(qm + (ss - vdm)*(rm*ss + pm/(sm - vdm))) - qm;
    const Real q_deltap  = chi_starp*(qp + (ss - vdp)*(rp*ss + pp/(sp - vdp))) - qp;

    // 3.)
    const Real fm = vdm*(qm + pm);
    const Real fp = vdp*(qp + pp);

    // 4.)
    const Real flux = (0.5f*(1.0f + sign_star)) * (fm + s_minus*q_deltam) + (0.5f*(1.0f - sign_star)) * (fp + s_pluss*q_deltap);
    assert(!isnan(flux));
    return flux;
}


__device__
inline Real _extraterm_hllc_vel(const Real um, const Real up,
        const Real Gm, const Real Gp,
        const Real Pm, const Real Pp,
        const Real sm, const Real sp, const Real ss) // (17 MUL/ADD/SUB + 2 DIV) = 19 FLOP
{
    const Real sign_star = (ss == 0.0f) ? 0.0f : ((ss < 0.0f) ? -1.0f : 1.0f);
    const Real s_minus   = fminf(0.0f, sm);
    const Real s_pluss   = fmaxf(0.0f, sp);
    const Real chi_starm = (sm - um)/(sm - ss) - 1.0f;
    const Real chi_starp = (sp - up)/(sp - ss) - 1.0f;

    return (0.5f*(1.0f + sign_star))*(um + s_minus*chi_starm) + (0.5f*(1.0f - sign_star))*(up + s_pluss*chi_starp);
}


///////////////////////////////////////////////////////////////////////////////
//                                  KERNELS                                  //
///////////////////////////////////////////////////////////////////////////////
__global__
void _xextraterm_hllc(const uint_t nslices, DevicePointer divF, DevicePointer flux,
        const Real * const __restrict__ Gm, const Real * const __restrict__ Gp,
        const Real * const __restrict__ Pm, const Real * const __restrict__ Pp,
        const Real * const __restrict__ vel,
        Real * const __restrict__ sumG, Real * const __restrict__ sumP, Real * const __restrict__ divU)
{
    const uint_t ix = blockIdx.x * _TILE_DIM_ + threadIdx.x;
    const uint_t iy = blockIdx.y * _TILE_DIM_ + threadIdx.y;

    // limiting resource
    __shared__ Real smem1[_TILE_DIM_][_TILE_DIM_+1];
    __shared__ Real smem2[_TILE_DIM_][_TILE_DIM_+1];
    __shared__ Real smem3[_TILE_DIM_][_TILE_DIM_+1];
    __shared__ Real smem4[_TILE_DIM_][_TILE_DIM_+1];
    __shared__ Real smem5[_TILE_DIM_][_TILE_DIM_+1];
    __shared__ Real smem6[_TILE_DIM_][_TILE_DIM_+1];

    if (ix < NX && iy < NY)
    {
        // transpose
        const uint_t iyT = blockIdx.y * _TILE_DIM_ + threadIdx.x;
        const uint_t ixT = blockIdx.x * _TILE_DIM_ + threadIdx.y;

        for (uint_t iz = 0; iz < nslices; ++iz)
        {
            for (int i = 0; i < _TILE_DIM_; i += _BLOCK_ROWS_)
            {
                const uint_t idxm = ID3(iyT,ixT+i,iz,NY,NXP1);
                const uint_t idxp = ID3(iyT,(ixT+1)+i,iz,NY,NXP1);

                // pre-fetch
                Real _sumG = Gp[idxm];
                Real _sumP = Pp[idxm];
                Real _divU = vel[idxp];
                _sumG = _sumG + Gm[idxp];
                _sumP = _sumP + Pm[idxp];
                _divU = _divU - vel[idxm];
                // read first batch
                smem1[threadIdx.x][threadIdx.y+i] = _sumG;
                smem2[threadIdx.x][threadIdx.y+i] = _sumP;
                smem3[threadIdx.x][threadIdx.y+i] = _divU;
            }
            __syncthreads();

            for (int i = 0; i < _TILE_DIM_; i += _BLOCK_ROWS_)
            {
                const uint_t idxm = ID3(iyT,ixT+i,iz,NY,NXP1);
                const uint_t idxp = ID3(iyT,(ixT+1)+i,iz,NY,NXP1);
                const uint_t idx = ID3(ix,iy+i,iz,NX,NY);

                // pre-fetch
                Real _divFr = flux.r[idxp];
                Real _divFu = flux.u[idxp];
                Real _divFv = flux.v[idxp];
                _divFr = _divFr - flux.r[idxm];
                _divFu = _divFu - flux.u[idxm];
                _divFv = _divFv - flux.v[idxm];
                // write first batch
                sumG[idx] = smem1[threadIdx.y+i][threadIdx.x];
                sumP[idx] = smem2[threadIdx.y+i][threadIdx.x];
                divU[idx] = smem3[threadIdx.y+i][threadIdx.x];
                // read second batch
                smem4[threadIdx.x][threadIdx.y+i] = _divFr;
                smem5[threadIdx.x][threadIdx.y+i] = _divFu;
                smem6[threadIdx.x][threadIdx.y+i] = _divFv;
            }
            __syncthreads();

            for (int i = 0; i < _TILE_DIM_; i += _BLOCK_ROWS_)
            {
                const uint_t idxm = ID3(iyT,ixT+i,iz,NY,NXP1);
                const uint_t idxp = ID3(iyT,(ixT+1)+i,iz,NY,NXP1);
                const uint_t idx = ID3(ix,iy+i,iz,NX,NY);

                // pre-fetch
                Real _divFw = flux.w[idxp];
                Real _divFe = flux.e[idxp];
                Real _divFG = flux.G[idxp];
                _divFw = _divFw - flux.w[idxm];
                _divFe = _divFe - flux.e[idxm];
                _divFG = _divFG - flux.G[idxm];
                // write second batch
                divF.r[idx] = smem4[threadIdx.y+i][threadIdx.x];
                divF.u[idx] = smem5[threadIdx.y+i][threadIdx.x];
                divF.v[idx] = smem6[threadIdx.y+i][threadIdx.x];
                // read third batch
                smem1[threadIdx.x][threadIdx.y+i] = _divFw;
                smem2[threadIdx.x][threadIdx.y+i] = _divFe;
                smem3[threadIdx.x][threadIdx.y+i] = _divFG;
            }
            __syncthreads();

            for (int i = 0; i < _TILE_DIM_; i += _BLOCK_ROWS_)
            {
                const uint_t idxm = ID3(iyT,ixT+i,iz,NY,NXP1);
                const uint_t idxp = ID3(iyT,(ixT+1)+i,iz,NY,NXP1);
                const uint_t idx = ID3(ix,iy+i,iz,NX,NY);

                // pre-fetch
                Real _divFP = flux.P[idxp];
                _divFP = _divFP - flux.P[idxm];
                // write third batch
                divF.w[idx] = smem1[threadIdx.y+i][threadIdx.x];
                divF.e[idx] = smem2[threadIdx.y+i][threadIdx.x];
                divF.G[idx] = smem3[threadIdx.y+i][threadIdx.x];
                // read fourth batch
                smem4[threadIdx.x][threadIdx.y+i] = _divFP;
            }
            __syncthreads();

            for (int i = 0; i < _TILE_DIM_; i += _BLOCK_ROWS_)
            {
                const uint_t idx = ID3(ix,iy+i,iz,NX,NY);
                // write fourth batch
                divF.P[idx] = smem4[threadIdx.y+i][threadIdx.x];
            }
            // NOTE: __syncthreads() can be omitted since it will not be
            // touched until next synchronization point
        }
    }
}


__global__
void _yextraterm_hllc(const uint_t nslices, DevicePointer divF, DevicePointer flux,
        const Real * const __restrict__ Gm, const Real * const __restrict__ Gp,
        const Real * const __restrict__ Pm, const Real * const __restrict__ Pp,
        const Real * const __restrict__ vel,
        Real * const __restrict__ sumG, Real * const __restrict__ sumP, Real * const __restrict__ divU)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        for (uint_t iz = 0; iz < nslices; ++iz)
        {
            const uint_t idx  = ID3(ix,iy,iz,NX,NY);
            const uint_t idxm = ID3(ix,iy,iz,NX,NYP1);
            const uint_t idxp = ID3(ix,(iy+1),iz,NX,NYP1);

            Real _sumG = Gp[idxm];
            Real _sumP = Pp[idxm];
            Real _divU = vel[idxp];
            Real _divFr = flux.r[idxp];
            Real _divFu = flux.u[idxp];
            Real _divFv = flux.v[idxp];
            Real _divFw = flux.w[idxp];
            Real _divFe = flux.e[idxp];
            Real _divFG = flux.G[idxp];
            Real _divFP = flux.P[idxp];
            _sumG = _sumG + Gm[idxp];
            _sumP = _sumP + Pm[idxp];
            _divU = _divU - vel[idxm];
            _divFr = _divFr - flux.r[idxm];
            _divFu = _divFu - flux.u[idxm];
            _divFv = _divFv - flux.v[idxm];
            _divFw = _divFw - flux.w[idxm];
            _divFe = _divFe - flux.e[idxm];
            _divFG = _divFG - flux.G[idxm];
            _divFP = _divFP - flux.P[idxm];

            sumG[idx] += _sumG;
            sumP[idx] += _sumP;
            divU[idx] += _divU;
            divF.r[idx] += _divFr;
            divF.u[idx] += _divFu;
            divF.v[idx] += _divFv;
            divF.w[idx] += _divFw;
            divF.e[idx] += _divFe;
            divF.G[idx] += _divFG;
            divF.P[idx] += _divFP;
        }
    }
}


__global__
void _zextraterm_hllc(const uint_t nslices, DevicePointer divF, DevicePointer flux,
        const Real * const __restrict__ Gm, const Real * const __restrict__ Gp,
        const Real * const __restrict__ Pm, const Real * const __restrict__ Pp,
        const Real * const __restrict__ vel,
        Real * const __restrict__ sumG, Real * const __restrict__ sumP, Real * const __restrict__ divU)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        for (uint_t iz = 0; iz < nslices; ++iz)
        {
            const uint_t idx  = ID3(ix,iy,iz,NX,NY);
            const uint_t idxm = ID3(ix,iy,iz,NX,NY);
            const uint_t idxp = ID3(ix,iy,(iz+1),NX,NY);

            const Real inv6 = 1.0f/6.0f;

            // cummulative sums of x and y
            const Real cumm_sumG = sumG[idx];
            const Real cumm_sumP = sumP[idx];
            const Real cumm_divU = divU[idx];

            Real _sumG = Gp[idxm];
            Real _sumP = Pp[idxm];
            Real _divU = vel[idxp];
            Real _divFr = flux.r[idxp];
            Real _divFu = flux.u[idxp];
            Real _divFv = flux.v[idxp];
            Real _divFw = flux.w[idxp];
            Real _divFe = flux.e[idxp];
            Real _divFG = flux.G[idxp];
            Real _divFP = flux.P[idxp];
            _sumG = _sumG + Gm[idxp] + cumm_sumG;
            _sumP = _sumP + Pm[idxp] + cumm_sumP;
            _divU = _divU - vel[idxm]+ cumm_divU;
            _divFr = _divFr - flux.r[idxm];
            _divFu = _divFu - flux.u[idxm];
            _divFv = _divFv - flux.v[idxm];
            _divFw = _divFw - flux.w[idxm];
            _divFe = _divFe - flux.e[idxm];
            _divFG = _divFG - flux.G[idxm];
            _divFP = _divFP - flux.P[idxm];

            // final divF
            divF.r[idx] += _divFr;
            divF.u[idx] += _divFu;
            divF.v[idx] += _divFv;
            divF.w[idx] += _divFw;
            divF.e[idx] += _divFe;
            divF.G[idx] += _divFG - inv6*_divU*_sumG;
            divF.P[idx] += _divFP - inv6*_divU*_sumP;
        }
    }
}


__global__
void _xflux00(const uint_t nslices, const uint_t global_iz,
        const DevicePointer ghostL, const DevicePointer ghostR, DevicePointer flux,
        Real * const __restrict__ xtra_vel,
        Real * const __restrict__ xtra_Gm, Real * const __restrict__ xtra_Gp,
        Real * const __restrict__ xtra_Pm, Real * const __restrict__ xtra_Pp)
{
    /* *
     * Notes:
     * ======
     * 1.) NXP1 = NX + 1
     * 2.) NX = NodeBlock::sizeX
     * 3.) NY = NodeBlock::sizeY
     * 4.) nslices = number of slices for currently processed chunk
     * 5.) global_iz is the iz-coordinate in index space of the NodeBlock for
     *     the first slice of the currently processed chunk.  It is needed if
     *     all of the x-/yghosts are uploaded to the GPU prior to processing
     *     the chunks sequentially.  Currently global_iz = 0, since x-/yghosts
     *     are uploaded per chunk.
     * 6.) Reads textures 00
     * */
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    assert(NXP1 > 5);

    if (ix < NXP1 && iy < NY)
    {
        for (uint_t iz = 3; iz < nslices+3; ++iz) // first and last 3 slices are zghosts
        {
            /* *
             * The general task order is (for each chunk slice along NZ):
             * 1.) Load stencils
             * 2.) Reconstruct primitive values using WENO5/WENO3
             * 3.) Compute characteristic velocities
             * 4.) Compute fluxes
             * 5.) Compute RHS for advection of G and P
             * */

            // stencils (7 * _STENCIL_WIDTH_ registers per thread)
            Real r[6];
            Real u[6];
            Real v[6];
            Real w[6];
            Real e[6];
            Real G[6];
            Real P[6];

            // 1.)
            // GMEM transactions are cached, effective GMEM accesses are 7*3
            // (according to nvvp)
            if (0 == ix)
                _load_3X00<0,0,3,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(3*GMEM + 3*tex_start)
            else if (1 == ix)
                _load_2X00<0,0,2,1>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(2*GMEM + 4*tex_start)
            else if (2 == ix)
                _load_1X00<0,0,1,2>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(1*GMEM + 5*tex_start)
            else if (NXP1-3 == ix)
                _load_1X00<NXP1-6,5,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NXP1-2 == ix)
                _load_2X00<NXP1-5,4,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NXP1-1 == ix)
                _load_3X00<NXP1-4,3,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else
                _load_internal_X00(ix, iy, iz, r, u, v, w, e, G, P, global_iz, NULL); // load 7*(6*tex_start)

            // compute body
#           include "xflux_body.cu"

            /* if (global_iz) */
            /* { */
/* #pragma unroll 6 */
            /*     for (uint_t i = 0; i < 6; ++i) */
            /*     { */
            /*         r[0] += r[i]; */
            /*         u[0] += u[i]; */
            /*         v[0] += v[i]; */
            /*         w[0] += w[i]; */
            /*         e[0] += e[i]; */
            /*         G[0] += G[i]; */
            /*         P[0] += P[i]; */
            /*     } */
            /* } */
            /* const uint_t idx = ID3(iy, ix, iz-3, NY, NXP1); */
            /* flux.r[idx] = r[0]; */
            /* flux.u[idx] = u[0]; */
            /* flux.v[idx] = v[0]; */
            /* flux.w[idx] = w[0]; */
            /* flux.e[idx] = e[0]; */
            /* flux.G[idx] = G[0]; */
            /* flux.P[idx] = P[0]; */
            /* xtra_vel[idx] = r[0]; */
            /* xtra_Gm[idx]  = w[0]; */
            /* xtra_Gp[idx]  = e[0]; */
            /* xtra_Pm[idx]  = P[0]; */
            /* xtra_Pp[idx]  = u[0]; */

            const uint_t idx = ID3(iy, ix, iz-3, NY, NXP1);
            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            // 5.)
            xtra_vel[idx] = hllc_vel;
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
}


__global__
void _xflux01(const uint_t nslices, const uint_t global_iz,
        const DevicePointer ghostL, const DevicePointer ghostR, DevicePointer flux,
        Real * const __restrict__ xtra_vel,
        Real * const __restrict__ xtra_Gm, Real * const __restrict__ xtra_Gp,
        Real * const __restrict__ xtra_Pm, Real * const __restrict__ xtra_Pp)
{
    /* *
     * Notes:
     * ======
     * 1.) NXP1 = NX + 1
     * 2.) NX = NodeBlock::sizeX
     * 3.) NY = NodeBlock::sizeY
     * 4.) nslices = number of slices for currently processed chunk
     * 5.) global_iz is the iz-coordinate in index space of the NodeBlock for
     *     the first slice of the currently processed chunk.  It is needed if
     *     all of the x-/yghosts are uploaded to the GPU prior to processing
     *     the chunks sequentially.  Currently global_iz = 0, since x-/yghosts
     *     are uploaded per chunk.
     * 6.) Reads textures 01
     * */
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    assert(NXP1 > 5);

    if (ix < NXP1 && iy < NY)
    {
        for (uint_t iz = 3; iz < nslices+3; ++iz) // first and last 3 slices are zghosts
        {
            /* *
             * The general task order is (for each chunk slice along NZ):
             * 1.) Load stencils
             * 2.) Reconstruct primitive values using WENO5/WENO3
             * 3.) Compute characteristic velocities
             * 4.) Compute fluxes
             * 5.) Compute RHS for advection of G and P
             * */

            // stencils (7 * _STENCIL_WIDTH_ registers per thread)
            Real r[6];
            Real u[6];
            Real v[6];
            Real w[6];
            Real e[6];
            Real G[6];
            Real P[6];

            // 1.)
            // GMEM transactions are cached, effective GMEM accesses are 7*3
            // (according to nvvp)
            if (0 == ix)
                _load_3X01<0,0,3,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(3*GMEM + 3*tex_start)
            else if (1 == ix)
                _load_2X01<0,0,2,1>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(2*GMEM + 4*tex_start)
            else if (2 == ix)
                _load_1X01<0,0,1,2>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(1*GMEM + 5*tex_start)
            else if (NXP1-3 == ix)
                _load_1X01<NXP1-6,5,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NXP1-2 == ix)
                _load_2X01<NXP1-5,4,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NXP1-1 == ix)
                _load_3X01<NXP1-4,3,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else
                _load_internal_X01(ix, iy, iz, r, u, v, w, e, G, P, global_iz, NULL); // load 7*(6*tex_start)

            // compute body
#           include "xflux_body.cu"

            const uint_t idx = ID3(iy, ix, iz-3, NY, NXP1);
            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            // 5.)
            xtra_vel[idx] = hllc_vel;
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
}


__global__
void _yflux00(const uint_t nslices, const uint_t global_iz,
        const DevicePointer ghostL, const DevicePointer ghostR, DevicePointer flux,
        Real * const __restrict__ xtra_vel,
        Real * const __restrict__ xtra_Gm, Real * const __restrict__ xtra_Gp,
        Real * const __restrict__ xtra_Pm, Real * const __restrict__ xtra_Pp)
{
    /* *
     * Notes:
     * ======
     * 1.) NYP1 = NY + 1
     * 2.) NX = NodeBlock::sizeX
     * 3.) NY = NodeBlock::sizeY
     * 4.) nslices = number of slices for currently processed chunk
     * 5.) global_iz is the iz-coordinate in index space of the NodeBlock for
     *     the first slice of the currently processed chunk.  It is needed if
     *     all of the x-/yghosts are uploaded to the GPU prior to processing
     *     the chunks sequentially.  Currently global_iz = 0, since x-/yghosts
     *     are uploaded per chunk.
     * 6.) Reads texture 00
     * */
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    assert(NYP1 > 5);

    if (ix < NX && iy < NYP1)
    {
        for (uint_t iz = 3; iz < nslices+3; ++iz) // first and last 3 slices are zghosts
        {
            /* *
             * The general task order is (for each chunk slice along NZ):
             * 1.) Load stencils
             * 2.) Reconstruct primitive values using WENO5/WENO3
             * 3.) Compute characteristic velocities
             * 4.) Compute fluxes
             * 5.) Compute RHS for advection of G and P
             * */

            // stencils (7 * _STENCIL_WIDTH_ registers per thread)
            Real r[6];
            Real u[6];
            Real v[6];
            Real w[6];
            Real e[6];
            Real G[6];
            Real P[6];

            // 1.)
            // GMEM transactions are cached, effective GMEM accesses are 7*3
            // (according to nvvp)
            if (0 == iy)
                _load_3Y00<0,0,3,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(3*GMEM + 3*TEX)
            else if (1 == iy)
                _load_2Y00<0,0,2,1>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(2*GMEM + 4*TEX)
            else if (2 == iy)
                _load_1Y00<0,0,1,2>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(1*GMEM + 5*TEX)
            else if (NYP1-3 == iy)
                _load_1Y00<NYP1-6,5,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NYP1-2 == iy)
                _load_2Y00<NYP1-5,4,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NYP1-1 == iy)
                _load_3Y00<NYP1-4,3,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else
                _load_internal_Y00(ix, iy, iz, r, u, v, w, e, G, P, global_iz, NULL); // load 7*(6*TEX)

            // compute body
#           include "yflux_body.cu"

            /* if (global_iz) */
            /* { */
/* #pragma unroll 6 */
            /*     for (uint_t i = 0; i < 6; ++i) */
            /*     { */
            /*         r[0] += r[i]; */
            /*         u[0] += u[i]; */
            /*         v[0] += v[i]; */
            /*         w[0] += w[i]; */
            /*         e[0] += e[i]; */
            /*         G[0] += G[i]; */
            /*         P[0] += P[i]; */
            /*     } */
            /* } */
            /* const uint_t idx = ID3(ix, iy, iz-3, NX, NYP1); */
            /* flux.r[idx] = r[0]; */
            /* flux.u[idx] = u[0]; */
            /* flux.v[idx] = v[0]; */
            /* flux.w[idx] = w[0]; */
            /* flux.e[idx] = e[0]; */
            /* flux.G[idx] = G[0]; */
            /* flux.P[idx] = P[0]; */
            /* xtra_vel[idx] = r[0]; */
            /* xtra_Gm[idx]  = w[0]; */
            /* xtra_Gp[idx]  = e[0]; */
            /* xtra_Pm[idx]  = P[0]; */
            /* xtra_Pp[idx]  = u[0]; */

            const uint_t idx = ID3(ix, iy, iz-3, NX, NYP1);
            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            // 5.)
            xtra_vel[idx] = hllc_vel;
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
}


__global__
void _yflux01(const uint_t nslices, const uint_t global_iz,
        const DevicePointer ghostL, const DevicePointer ghostR, DevicePointer flux,
        Real * const __restrict__ xtra_vel,
        Real * const __restrict__ xtra_Gm, Real * const __restrict__ xtra_Gp,
        Real * const __restrict__ xtra_Pm, Real * const __restrict__ xtra_Pp)
{
    /* *
     * Notes:
     * ======
     * 1.) NYP1 = NY + 1
     * 2.) NX = NodeBlock::sizeX
     * 3.) NY = NodeBlock::sizeY
     * 4.) nslices = number of slices for currently processed chunk
     * 5.) global_iz is the iz-coordinate in index space of the NodeBlock for
     *     the first slice of the currently processed chunk.  It is needed if
     *     all of the x-/yghosts are uploaded to the GPU prior to processing
     *     the chunks sequentially.  Currently global_iz = 0, since x-/yghosts
     *     are uploaded per chunk.
     * 6.) Reads texture 01
     * */
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    assert(NYP1 > 5);

    if (ix < NX && iy < NYP1)
    {
        for (uint_t iz = 3; iz < nslices+3; ++iz) // first and last 3 slices are zghosts
        {
            /* *
             * The general task order is (for each chunk slice along NZ):
             * 1.) Load stencils
             * 2.) Reconstruct primitive values using WENO5/WENO3
             * 3.) Compute characteristic velocities
             * 4.) Compute fluxes
             * 5.) Compute RHS for advection of G and P
             * */

            // stencils (7 * _STENCIL_WIDTH_ registers per thread)
            Real r[6];
            Real u[6];
            Real v[6];
            Real w[6];
            Real e[6];
            Real G[6];
            Real P[6];

            // 1.)
            // GMEM transactions are cached, effective GMEM accesses are 7*3
            // (according to nvvp)
            if (0 == iy)
                _load_3Y01<0,0,3,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(3*GMEM + 3*TEX)
            else if (1 == iy)
                _load_2Y01<0,0,2,1>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(2*GMEM + 4*TEX)
            else if (2 == iy)
                _load_1Y01<0,0,1,2>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(1*GMEM + 5*TEX)
            else if (NYP1-3 == iy)
                _load_1Y01<NYP1-6,5,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NYP1-2 == iy)
                _load_2Y01<NYP1-5,4,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NYP1-1 == iy)
                _load_3Y01<NYP1-4,3,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else
                _load_internal_Y01(ix, iy, iz, r, u, v, w, e, G, P, global_iz, NULL); // load 7*(6*TEX)

            // compute body
#           include "yflux_body.cu"

            const uint_t idx = ID3(ix, iy, iz-3, NX, NYP1);
            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            // 5.)
            xtra_vel[idx] = hllc_vel;
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
}


__global__
void _zflux00(const uint_t nslices, DevicePointer flux,
        Real * const __restrict__ xtra_vel,
        Real * const __restrict__ xtra_Gm, Real * const __restrict__ xtra_Gp,
        Real * const __restrict__ xtra_Pm, Real * const __restrict__ xtra_Pp)
{
    /* *
     * Notes:
     * ======
     * 1.) NX = NodeBlock::sizeX
     * 2.) NY = NodeBlock::sizeY
     * 3.) NZ = NodeBlock::sizeZ
     * 4.) nslices = number of slices for currently processed chunk
     * 5.) Reads texture 00
     * */
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    // depends on boundary condition in z-direction
    assert(NodeBlock::sizeZ > 0);

    if (ix < NX && iy < NY)
    {
        for (uint_t iz = 3; iz < (nslices+1)+3; ++iz) // first and last 3 slices are zghosts; need to compute nslices+1 fluxes in z-direction
        {
            /* *
             * The general task order is (for each chunk slice along NZ):
             * 1.) Load stencils
             * 2.) Reconstruct primitive values using WENO5/WENO3
             * 3.) Compute characteristic velocities
             * 4.) Compute fluxes
             * 5.) Compute RHS for advection of G and P
             * */

            // stencils (7 * _STENCIL_WIDTH_ registers per thread)
            Real r[6];
            Real u[6];
            Real v[6];
            Real w[6];
            Real e[6];
            Real G[6];
            Real P[6];

            // 1.)
            _load_internal_Z00(ix, iy, iz, r, u, v, w, e, G, P, 0, NULL); // load 7*(6*TEX)

            // compute body
#           include "zflux_body.cu"

            /* if (global_iz) */
            /* { */
/* #pragma unroll 6 */
            /*     for (uint_t i = 0; i < 6; ++i) */
            /*     { */
            /*         r[0] += r[i]; */
            /*         u[0] += u[i]; */
            /*         v[0] += v[i]; */
            /*         w[0] += w[i]; */
            /*         e[0] += e[i]; */
            /*         G[0] += G[i]; */
            /*         P[0] += P[i]; */
            /*     } */
            /* } */
            /* const uint_t idx = ID3(ix, iy, iz-3, NX, NY); */
            /* flux.r[idx] = r[0]; */
            /* flux.u[idx] = u[0]; */
            /* flux.v[idx] = v[0]; */
            /* flux.w[idx] = w[0]; */
            /* flux.e[idx] = e[0]; */
            /* flux.G[idx] = G[0]; */
            /* flux.P[idx] = P[0]; */
            /* xtra_vel[idx] = r[0]; */
            /* xtra_Gm[idx]  = w[0]; */
            /* xtra_Gp[idx]  = e[0]; */
            /* xtra_Pm[idx]  = P[0]; */
            /* xtra_Pp[idx]  = u[0]; */

            const uint_t idx = ID3(ix, iy, iz-3, NX, NY);
            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            // 5.)
            xtra_vel[idx] = hllc_vel;
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
}


__global__
void _zflux01(const uint_t nslices, DevicePointer flux,
        Real * const __restrict__ xtra_vel,
        Real * const __restrict__ xtra_Gm, Real * const __restrict__ xtra_Gp,
        Real * const __restrict__ xtra_Pm, Real * const __restrict__ xtra_Pp)
{
    /* *
     * Notes:
     * ======
     * 1.) NX = NodeBlock::sizeX
     * 2.) NY = NodeBlock::sizeY
     * 3.) NZ = NodeBlock::sizeZ
     * 4.) nslices = number of slices for currently processed chunk
     * 5.) Reads texture 01
     * */
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    // depends on boundary condition in z-direction
    assert(NodeBlock::sizeZ > 0);

    if (ix < NX && iy < NY)
    {
        for (uint_t iz = 3; iz < (nslices+1)+3; ++iz) // first and last 3 slices are zghosts; need to compute nslices+1 fluxes in z-direction
        {
            /* *
             * The general task order is (for each chunk slice along NZ):
             * 1.) Load stencils
             * 2.) Reconstruct primitive values using WENO5/WENO3
             * 3.) Compute characteristic velocities
             * 4.) Compute fluxes
             * 5.) Compute RHS for advection of G and P
             * */

            // stencils (7 * _STENCIL_WIDTH_ registers per thread)
            Real r[6];
            Real u[6];
            Real v[6];
            Real w[6];
            Real e[6];
            Real G[6];
            Real P[6];

            // 1.)
            _load_internal_Z01(ix, iy, iz, r, u, v, w, e, G, P, 0, NULL); // load 7*(6*TEX)

            // compute body
#           include "zflux_body.cu"

            const uint_t idx = ID3(ix, iy, iz-3, NX, NY);
            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            // 5.)
            xtra_vel[idx] = hllc_vel;
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
}


__global__
void _maxSOS(const uint_t nslices, int* g_maxSOS)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    const uint_t loc_idx = blockDim.x * threadIdx.y + threadIdx.x;
    __shared__ Real block_sos[_NTHREADS_];
    block_sos[loc_idx] = 0.0f;

    if (ix < NX && iy < NY)
    {
        Real sos = 0.0f;

        for (uint_t iz = 0; iz < nslices; ++iz)
        {
            // TODO: used both buffers here
            const Real r = tex3D(texR00, ix, iy, iz);
            const Real u = tex3D(texU00, ix, iy, iz);
            const Real v = tex3D(texV00, ix, iy, iz);
            const Real w = tex3D(texW00, ix, iy, iz);
            const Real e = tex3D(texE00, ix, iy, iz);
            const Real G = tex3D(texG00, ix, iy, iz);
            const Real P = tex3D(texP00, ix, iy, iz);

            const Real p = (e - (u*u + v*v + w*w)*(0.5f/r) - P) / G;
            const Real c = sqrtf(((p + P) / G + p) / r);

            sos = fmaxf(sos, c + fmaxf(fmaxf(fabsf(u), fabsf(v)), fabsf(w)) / r);
        }
        block_sos[loc_idx] = sos;
        __syncthreads();

        if (0 == loc_idx)
        {
            for (int i = 1; i < _NTHREADS_; ++i)
                sos = fmaxf(sos, block_sos[i]);
            assert(sos > 0.0f);
            atomicMax(g_maxSOS, __float_as_int(sos));
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
//                              KERNEL WRAPPERS                              //
///////////////////////////////////////////////////////////////////////////////
static void _bindTexture(texture<float, 3, cudaReadModeElementType> * const tex, cudaArray_t d_ptr)
{
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<Real>();
    tex->addressMode[0]       = cudaAddressModeClamp;
    tex->addressMode[1]       = cudaAddressModeClamp;
    tex->addressMode[2]       = cudaAddressModeClamp;
    tex->channelDesc          = fmt;
    tex->filterMode           = cudaFilterModePoint;
    tex->mipmapFilterMode     = cudaFilterModePoint;
    tex->normalized           = false;

    cudaBindTextureToArray(tex, d_ptr, &fmt);
}


void GPU::compute_pipe_divF(const uint_t nslices, const uint_t global_iz,
        const uint_t gbuf_id, const int chunk_id)
{
#ifndef _MUTE_GPU_
    assert(gbuf_id < _NUM_GPU_BUF_);

    /* *
     * Compute div(F)
     * */

    // my stream
    const uint_t s_id = chunk_id % _NUM_STREAMS_;

    // my data
    GPU_COMM * const mybuf = &gpu_comm[gbuf_id];

    // my ghosts
    DevicePointer xghostL(mybuf->d_xgl);
    DevicePointer xghostR(mybuf->d_xgr);
    DevicePointer yghostL(mybuf->d_ygl);
    DevicePointer yghostR(mybuf->d_ygr);

    // my output
    DevicePointer divF(mybuf->d_divF);

    // my tmp storage
    DevicePointer flux(d_flux);

    // my launch config
    const dim3 X_blocks(1, _NTHREADS_, 1);
    const dim3 X_grid(NXP1, (NY + _NTHREADS_ -1)/_NTHREADS_, 1);
    const dim3 X_xtraBlocks(_TILE_DIM_, _BLOCK_ROWS_, 1);
    const dim3 X_xtraGrid((NX + _TILE_DIM_ - 1)/_TILE_DIM_, (NY + _TILE_DIM_ - 1)/_TILE_DIM_, 1);

    const dim3 Y_blocks(_NTHREADS_, 1, 1);
    const dim3 Y_grid((NX + _NTHREADS_ -1) / _NTHREADS_, NYP1, 1);
    const dim3 Y_xtraGrid((NX + _NTHREADS_ -1) / _NTHREADS_, NY, 1);

    const dim3 Z_blocks(_NTHREADS_, 1, 1);
    const dim3 Z_grid((NX + _NTHREADS_ -1) / _NTHREADS_, NY, 1);

    // previous stream has priority. Since resources are limited, concurrent
    // kernels make less sense
    const uint_t s_idm1 = ((chunk_id-1) + _NUM_STREAMS_) % _NUM_STREAMS_;
    assert(s_idm1 < _NUM_STREAMS_);
    cudaStreamWaitEvent(stream[s_id], event_compute[s_idm1], 0);

    char prof_item[256];

    // queue kernels in pipe
    switch (gbuf_id)
    {
        case 0:
            _bindTexture(&texR00, mybuf->d_GPUin[0]);
            _bindTexture(&texU00, mybuf->d_GPUin[1]);
            _bindTexture(&texV00, mybuf->d_GPUin[2]);
            _bindTexture(&texW00, mybuf->d_GPUin[3]);
            _bindTexture(&texE00, mybuf->d_GPUin[4]);
            _bindTexture(&texG00, mybuf->d_GPUin[5]);
            _bindTexture(&texP00, mybuf->d_GPUin[6]);
            // --- X ---
            sprintf(prof_item, "_XFLUX (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _xflux00<<<X_grid, X_blocks, 0, stream[s_id]>>>(nslices, global_iz, xghostL, xghostR, flux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            GPU::profiler.pop_stopCUDA();

            sprintf(prof_item, "_XEXTRATERM (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _xextraterm_hllc<<<X_xtraGrid, X_xtraBlocks, 0, stream[s_id]>>>(nslices, divF, flux, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            GPU::profiler.pop_stopCUDA();

            // --- Y ---
            sprintf(prof_item, "_YFLUX (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _yflux00<<<Y_grid, Y_blocks, 0, stream[s_id]>>>(nslices, global_iz, yghostL, yghostR, flux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            GPU::profiler.pop_stopCUDA();

            sprintf(prof_item, "_YEXTRATERM (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _yextraterm_hllc<<<Y_xtraGrid, Y_blocks, 0, stream[s_id]>>>(nslices, divF, flux, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            GPU::profiler.pop_stopCUDA();

            // --- Z ---
            sprintf(prof_item, "_ZFLUX (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _zflux00<<<Z_grid, Z_blocks, 0, stream[s_id]>>>(nslices, flux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            GPU::profiler.pop_stopCUDA();

            sprintf(prof_item, "_ZEXTRATERM (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _zextraterm_hllc<<<Z_grid, Z_blocks, 0, stream[s_id]>>>(nslices, divF, flux, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            GPU::profiler.pop_stopCUDA();
            break;

        case 1:
            _bindTexture(&texR01, mybuf->d_GPUin[0]);
            _bindTexture(&texU01, mybuf->d_GPUin[1]);
            _bindTexture(&texV01, mybuf->d_GPUin[2]);
            _bindTexture(&texW01, mybuf->d_GPUin[3]);
            _bindTexture(&texE01, mybuf->d_GPUin[4]);
            _bindTexture(&texG01, mybuf->d_GPUin[5]);
            _bindTexture(&texP01, mybuf->d_GPUin[6]);
            // --- X ---
            sprintf(prof_item, "_XFLUX (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _xflux01<<<X_grid, X_blocks, 0, stream[s_id]>>>(nslices, global_iz, xghostL, xghostR, flux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            GPU::profiler.pop_stopCUDA();

            sprintf(prof_item, "_XEXTRATERM (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _xextraterm_hllc<<<X_xtraGrid, X_xtraBlocks, 0, stream[s_id]>>>(nslices, divF, flux, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            GPU::profiler.pop_stopCUDA();

            // --- Y ---
            sprintf(prof_item, "_YFLUX (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _yflux01<<<Y_grid, Y_blocks, 0, stream[s_id]>>>(nslices, global_iz, yghostL, yghostR, flux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            GPU::profiler.pop_stopCUDA();

            sprintf(prof_item, "_YEXTRATERM (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _yextraterm_hllc<<<Y_xtraGrid, Y_blocks, 0, stream[s_id]>>>(nslices, divF, flux, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            GPU::profiler.pop_stopCUDA();

            // --- Z ---
            sprintf(prof_item, "_ZFLUX (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _zflux01<<<Z_grid, Z_blocks, 0, stream[s_id]>>>(nslices, flux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            GPU::profiler.pop_stopCUDA();

            sprintf(prof_item, "_ZEXTRATERM (s_id=%d)", s_id);
            GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
            _zextraterm_hllc<<<Z_grid, Z_blocks, 0, stream[s_id]>>>(nslices, divF, flux, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            GPU::profiler.pop_stopCUDA();
            break;
    }

    cudaEventRecord(event_compute[s_id], stream[s_id]);
#endif
}


void GPU::MaxSpeedOfSound(const uint_t nslices, const uint_t gbuf_id, const int chunk_id)
{
#ifndef _MUTE_GPU_
    assert(gbuf_id < _NUM_GPU_BUF_);

    // my stream
    const uint_t s_id = chunk_id % _NUM_STREAMS_;

    // my data
    GPU_COMM * const mybuf = &gpu_comm[gbuf_id];

    _bindTexture(&texR00, mybuf->d_GPUin[0]);
    _bindTexture(&texU00, mybuf->d_GPUin[1]);
    _bindTexture(&texV00, mybuf->d_GPUin[2]);
    _bindTexture(&texW00, mybuf->d_GPUin[3]);
    _bindTexture(&texE00, mybuf->d_GPUin[4]);
    _bindTexture(&texG00, mybuf->d_GPUin[5]);
    _bindTexture(&texP00, mybuf->d_GPUin[6]);

    // my launch config
    const dim3 grid((NX + _NTHREADS_ -1) / _NTHREADS_, NY, 1);
    const dim3 blocks(_NTHREADS_, 1, 1);

    char prof_item[256];

    sprintf(prof_item, "_MAXSOS (s_id=%d)", s_id);
    GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
    _maxSOS<<<grid, blocks, 0, stream[s_id]>>>(nslices, d_maxSOS);
    GPU::profiler.pop_stopCUDA();
#endif
}

///////////////////////////////////////////////////////////////////////////
// TEST SECTION
///////////////////////////////////////////////////////////////////////////
void GPU::TestKernel()
{
    const uint_t gbuf_id = 0;
    const uint_t s_id = 0;

    // my data
    GPU_COMM * const mybuf = &gpu_comm[gbuf_id];

    _bindTexture(&texR00, mybuf->d_GPUin[0]);
    _bindTexture(&texU00, mybuf->d_GPUin[1]);
    _bindTexture(&texV00, mybuf->d_GPUin[2]);
    _bindTexture(&texW00, mybuf->d_GPUin[3]);
    _bindTexture(&texE00, mybuf->d_GPUin[4]);
    _bindTexture(&texG00, mybuf->d_GPUin[5]);
    _bindTexture(&texP00, mybuf->d_GPUin[6]);

    // my ghosts
    DevicePointer xghostL(mybuf->d_xgl);
    DevicePointer xghostR(mybuf->d_xgr);
    DevicePointer yghostL(mybuf->d_ygl);
    DevicePointer yghostR(mybuf->d_ygr);

    // my output
    DevicePointer divF(mybuf->d_divF);

    // my tmp storage
    DevicePointer flux(d_flux);

    cudaFree(d_Gm);
    cudaFree(d_Gp);
    cudaFree(d_Pm);
    cudaFree(d_Pp);
    cudaFree(d_hllc_vel);
    cudaFree(d_sumG);
    cudaFree(d_sumP);
    cudaFree(d_divU);

    const uint_t nslices = NodeBlock::sizeZ;
    const uint_t xflxSize = (NodeBlock::sizeX+1)*NodeBlock::sizeY*nslices;
    const uint_t yflxSize = NodeBlock::sizeX*(NodeBlock::sizeY+1)*nslices;
    const uint_t zflxSize = NodeBlock::sizeX*NodeBlock::sizeY*(nslices+1);

    Real *d_extra_X[5];
    Real *d_extra_Y[5];
    Real *d_extra_Z[5];
    for (int i = 0; i < 5; ++i)
    {
        cudaMalloc(&(d_extra_X[i]), xflxSize * sizeof(Real));
        cudaMalloc(&(d_extra_Y[i]), yflxSize * sizeof(Real));
        cudaMalloc(&(d_extra_Z[i]), zflxSize * sizeof(Real));
    }
    GPU::tell_memUsage_GPU();


    {

        const dim3 xblocks(1, _NTHREADS_, 1);
        const dim3 yblocks(_NTHREADS_, 1, 1);
        const dim3 zblocks(_NTHREADS_, 1, 1);
        const dim3 xgrid(NXP1, (NY + _NTHREADS_ - 1) / _NTHREADS_,   1);
        const dim3 ygrid((NX   + _NTHREADS_ - 1) / _NTHREADS_, NYP1, 1);
        const dim3 zgrid((NX   + _NTHREADS_ - 1) / _NTHREADS_, NY,   1);

        GPU::profiler.push_startCUDA("_XFLUX", &stream[s_id]);
        _xflux00<<<xgrid, xblocks, 0, stream[s_id]>>>(nslices, 0, xghostL, xghostR, flux, d_extra_X[0], d_extra_X[1], d_extra_X[2], d_extra_X[3], d_extra_X[4]);
        GPU::profiler.pop_stopCUDA();

        /* GPU::profiler.push_startCUDA("_YFLUX", &_s[0]); */
        /* _yflux<<<ygrid, yblocks, 0, _s[0]>>>(nslices, 0, yghostL, yghostR, flux, d_extra_Y[0], d_extra_Y[1], d_extra_Y[2], d_extra_Y[3], d_extra_Y[4]); */
        /* GPU::profiler.pop_stopCUDA(); */

        /* GPU::profiler.push_startCUDA("_ZFLUX", &_s[0]); */
        /* _zflux<<<zgrid, zblocks, 0, _s[0]>>>(nslices, flux, d_extra_Z[0], d_extra_Z[1], d_extra_Z[2], d_extra_Z[3], d_extra_Z[4]); */
        /* GPU::profiler.pop_stopCUDA(); */

        /* _xflux<<<xgrid, xblocks, 0, _s[0]>>>(nslices, 0, xghostL, xghostR, flux, d_extra_X[0], d_extra_X[1], d_extra_X[2], d_extra_X[3], d_extra_X[4]); */
        /* _yflux<<<ygrid, yblocks, 0, _s[1]>>>(nslices, 0, yghostL, yghostR, flux, d_extra_Y[0], d_extra_Y[1], d_extra_Y[2], d_extra_Y[3], d_extra_Y[4]); */
        /* _zflux<<<zgrid, zblocks, 0, _s[2]>>>(nslices, flux, d_extra_Z[0], d_extra_Z[1], d_extra_Z[2], d_extra_Z[3], d_extra_Z[4]); */

        cudaDeviceSynchronize();
    }
}

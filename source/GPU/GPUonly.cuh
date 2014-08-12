/* *
 * GPUonly.cuh
 *
 * Created by Fabian Wermelinger on 6/25/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include "GPU.h" // includes Types.h

#define NX _BLOCKSIZEX_
#define NY _BLOCKSIZEY_
#define NXP1 NX+1
#define NYP1 NY+1

struct DevicePointer // 7 fluid quantities
{
    // helper structure to pass compound flow variables as one kernel argument
    Real * __restrict__ r;
    Real * __restrict__ u;
    Real * __restrict__ v;
    Real * __restrict__ w;
    Real * __restrict__ e;
    Real * __restrict__ G;
    Real * __restrict__ P;
    DevicePointer(RealPtrVec_t& c) : r(c[0]), u(c[1]), v(c[2]), w(c[3]), e(c[4]), G(c[5]), P(c[6]) { assert(c.size() == 7); }
};


struct Stencil
{
    // stencil data compound centered around a face with ID i-1/2:
    // (i-3) (i-2) (i-1) | (i) (i+1) (i+2)
    // the face with ID i-1/2 is indicated by "|"
    Real im3, im2, im1, i, ip1, ip2;

    __device__
    Stencil(): im3(0), im2(0), im1(0), i(0), ip1(0), ip2(0) { }

    __device__
    inline bool operator>(const Real f) { return (im3>f && im2>f && im1>f && i>f && ip1>f && ip2>f); }
    __device__
    inline bool operator>=(const Real f) { return (im3>=f && im2>=f && im1>=f && i>=f && ip1>=f && ip2>=f); }
    __device__
    inline bool operator<(const Real f) { return (im3<f && im2<f && im1<f && i<f && ip1<f && ip2<f); }
    __device__
    inline bool operator<=(const Real f) { return (im3<=f && im2<=f && im1<=f && i<=f && ip1<=f && ip2<=f); }
};


///////////////////////////////////////////////////////////////////////////////
//                           GLOBAL VARIABLES                                //
///////////////////////////////////////////////////////////////////////////////
extern RealPtrVec_t d_tmp;
extern RealPtrVec_t d_rhs;
extern RealPtrVec_t d_xgl;
extern RealPtrVec_t d_xgr;
extern RealPtrVec_t d_ygl;
extern RealPtrVec_t d_ygr;

extern RealPtrVec_t d_xflux;
extern RealPtrVec_t d_yflux;
extern RealPtrVec_t d_zflux;

// 3D arrays
extern std::vector<cudaArray_t> d_GPUin;

// extraterms for advection equations
extern Real *d_Gm, *d_Gp;
extern Real *d_Pm, *d_Pp;
extern Real *d_hllc_vel;
extern Real *d_sumG, *d_sumP, *d_divU;

// max SOS
extern int *d_maxSOS;

// use non-null stream (async)
extern cudaStream_t *stream;

// TODO: remove
extern cudaStream_t stream1;
extern cudaStream_t stream2;

// events
extern cudaEvent_t *event;

// TODO: remove
extern cudaEvent_t h2d_tmp_completed;
extern cudaEvent_t divergence_completed;
extern cudaEvent_t update_completed;

// texture references
texture<float, 3, cudaReadModeElementType> texR;
texture<float, 3, cudaReadModeElementType> texU;
texture<float, 3, cudaReadModeElementType> texV;
texture<float, 3, cudaReadModeElementType> texW;
texture<float, 3, cudaReadModeElementType> texE;
texture<float, 3, cudaReadModeElementType> texG;
texture<float, 3, cudaReadModeElementType> texP;


///////////////////////////////////////////////////////////////////////////////
//                             DEVICE FUNCTIONS                              //
///////////////////////////////////////////////////////////////////////////////
__device__
inline void _weno_reconstruction(Real& recon_m, Real& recon_p, const Real * const __restrict__ s)
{
    const Real wenoeps_f = (Real)WENOEPS;

#ifndef _WENO3_
    // 184 FLOP
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

    const Real sum0m =  inv3*s[0] - 7.0f*inv6*s[1] + 11.0f*inv6*s[2];
    const Real sum1m = -inv6*s[1] + 5.0f*inv6*s[2] +       inv3*s[3];
    const Real sum2p =  inv3*s[3] + 5.0f*inv6*s[2] -       inv6*s[1];
    const Real sum2m =  inv3*s[2] + 5.0f*inv6*s[3] -       inv6*s[4];
    const Real sum1p = -inv6*s[4] + 5.0f*inv6*s[3] +       inv3*s[2];
    const Real sum0p =  inv3*s[5] - 7.0f*inv6*s[4] + 11.0f*inv6*s[3];

    const Real is0m = s[0]*(s[0]*q1 - s[1]*q2 + s[2]*q3) + s[1]*(s[1]*q4 - s[2]*q5) + s[2]*s[2]*q6;
    const Real is1m = s[1]*(s[1]*q1 - s[2]*q7 + s[3]*q8) + s[2]*(s[2]*q7 - s[3]*q7) + s[3]*s[3]*q1;
    const Real is2p = s[1]*(s[1]*q1 - s[2]*q2 + s[3]*q3) + s[2]*(s[2]*q4 - s[3]*q5) + s[3]*s[3]*q6;
    const Real is2m = s[2]*(s[2]*q6 - s[3]*q5 + s[4]*q3) + s[3]*(s[3]*q4 - s[4]*q2) + s[4]*s[4]*q1;
    const Real is1p = s[2]*(s[2]*q1 - s[3]*q7 + s[4]*q8) + s[3]*(s[3]*q7 - s[4]*q7) + s[4]*s[4]*q1;
    const Real is0p = s[3]*(s[3]*q6 - s[4]*q5 + s[5]*q3) + s[4]*(s[4]*q4 - s[5]*q2) + s[5]*s[5]*q1;

    const Real is0plusm = is0m + wenoeps_f;
    const Real is1plusm = is1m + wenoeps_f;
    const Real is2plusp = is2p + wenoeps_f;
    const Real is2plusm = is2m + wenoeps_f;
    const Real is1plusp = is1p + wenoeps_f;
    const Real is0plusp = is0p + wenoeps_f;

    const Real alpha0m = 1.0f / (10.0f*is0plusm*is0plusm);
    const Real alpha1m = 6.0f * (1.0f / (10.0f*is1plusm*is1plusm));
    const Real alpha2p = 3.0f * (1.0f / (10.0f*is2plusp*is2plusp));
    const Real alpha2m = 3.0f * (1.0f / (10.0f*is2plusm*is2plusm));
    const Real alpha1p = 6.0f * (1.0f / (10.0f*is1plusp*is1plusp));
    const Real alpha0p = 1.0f / (10.0f*is0plusp*is0plusp);
    const Real alphasumInvm = 1.0f / (alpha0m+alpha1m+alpha2m);
    const Real alphasumInvp = 1.0f / (alpha0p+alpha1p+alpha2p);

    const Real omega0m = alpha0m * alphasumInvm;
    const Real omega1m = alpha1m * alphasumInvm;
    const Real omega1p = alpha1p * alphasumInvp;
    const Real omega0p = alpha0p * alphasumInvp;
    const Real omega2m = 1.0f - omega0m - omega1m;
    const Real omega2p = 1.0f - omega0p - omega1p;

    recon_m = omega0m*sum0m + omega1m*sum1m + omega2m*sum2m;
    recon_p = omega0p*sum0p + omega1p*sum1p + omega2p*sum2p;

#else
    // WENO 3
    // 45 FLOP
    const Real sum0m  = 1.5f*s[2] - 0.5f*s[1];
    const Real sum1mp = 0.5f*(s[2] + s[3]);
    const Real sum0p  = 1.5f*s[3] - 0.5f*s[4];

    const Real is0m  = (s[2]-s[1])*(s[2]-s[1]);
    const Real is1mp = (s[3]-s[2])*(s[3]-s[2]);
    const Real is0p  = (s[3]-s[4])*(s[3]-s[4]);

    const Real alpha0m  = 1.0f / (3.0f * (is0m+wenoeps_f)*(is0m+wenoeps_f));
    const Real alpha1mp = 2.0f * (1.0f / (3.0f * (is1mp+wenoeps_f)*(is1mp+wenoeps_f)));
    const Real alpha0p  = 1.0f / (3.0f * (is0p+wenoeps_f)*(is0p+wenoeps_f));

    const Real omega0m = alpha0m / (alpha0m+alpha1mp);
    const Real omega0p = alpha0p / (alpha0p+alpha1mp);
    const Real omega1m = 1.0f - omega0m;
    const Real omega1p = 1.0f - omega0p;

    recon_m = omega0m*sum0m + omega1m*sum1mp;
    recon_p = omega0p*sum0p + omega1p*sum1mp;

#endif
}


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
inline Real _weno_clip_minus(const Real recon_m, const Real b, const Real c, const Real d)
{
    const Real min_in = fminf( fminf(b,c), d );
    const Real max_in = fmaxf( fmaxf(b,c), d );
    return fminf(fmaxf(recon_m, min_in), max_in);
}


__device__
inline Real _weno_clip_pluss(const Real recon_p, const Real c, const Real d, const Real e)
{
    const Real min_in = fminf( fminf(c,d), e );
    const Real max_in = fmaxf( fmaxf(c,d), e );
    return fminf(fmaxf(recon_p, min_in), max_in);
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


__device__
inline void _fetch_flux(const uint_t ix, const uint_t iy, const uint_t iz,
        const Real * const __restrict__ xflux, const Real * const __restrict__ yflux, const Real * const __restrict__ zflux,
        Real& fxp, Real& fxm, Real& fyp, Real& fym, Real& fzp, Real& fzm)
{
    /* fxp = xflux[ID3(ix+1, iy, iz, NXP1, NY)]; */
    /* fxm = xflux[ID3(ix,   iy, iz, NXP1, NY)]; */
    fxp = xflux[ID3(iy, ix+1, iz, NY, NXP1)];
    fxm = xflux[ID3(iy, ix,   iz, NY, NXP1)];

    fyp = yflux[ID3(ix, iy+1, iz, NX, NYP1)];
    fym = yflux[ID3(ix, iy,   iz, NX, NYP1)];

    fzp = zflux[ID3(ix, iy, iz+1, NX, NY)];
    fzm = zflux[ID3(ix, iy, iz,   NX, NY)];
    assert(!isnan(fxp));
    assert(!isnan(fxm));
    assert(!isnan(fyp));
    assert(!isnan(fym));
    assert(!isnan(fzp));
    assert(!isnan(fzm));
}


__device__
inline void _load_internal_X(const uint_t ix, const uint_t iy, const uint_t iz,
        Real * const __restrict__ r,
        Real * const __restrict__ u,
        Real * const __restrict__ v,
        Real * const __restrict__ w,
        Real * const __restrict__ e,
        Real * const __restrict__ G,
        Real * const __restrict__ P,
        const uint_t dummy1,
        const DevicePointer * const dummy2)
{
    // texture only
    r[0]  = tex3D(texR, ix-3, iy, iz);
    r[1]  = tex3D(texR, ix-2, iy, iz);
    r[2]  = tex3D(texR, ix-1, iy, iz);
    r[3]  = tex3D(texR, ix,   iy, iz);
    r[4]  = tex3D(texR, ix+1, iy, iz);
    r[5]  = tex3D(texR, ix+2, iy, iz);

    u[0]  = tex3D(texU, ix-3, iy, iz);
    u[1]  = tex3D(texU, ix-2, iy, iz);
    u[2]  = tex3D(texU, ix-1, iy, iz);
    u[3]  = tex3D(texU, ix,   iy, iz);
    u[4]  = tex3D(texU, ix+1, iy, iz);
    u[5]  = tex3D(texU, ix+2, iy, iz);

    v[0]  = tex3D(texV, ix-3, iy, iz);
    v[1]  = tex3D(texV, ix-2, iy, iz);
    v[2]  = tex3D(texV, ix-1, iy, iz);
    v[3]  = tex3D(texV, ix,   iy, iz);
    v[4]  = tex3D(texV, ix+1, iy, iz);
    v[5]  = tex3D(texV, ix+2, iy, iz);

    w[0]  = tex3D(texW, ix-3, iy, iz);
    w[1]  = tex3D(texW, ix-2, iy, iz);
    w[2]  = tex3D(texW, ix-1, iy, iz);
    w[3]  = tex3D(texW, ix,   iy, iz);
    w[4]  = tex3D(texW, ix+1, iy, iz);
    w[5]  = tex3D(texW, ix+2, iy, iz);

    e[0]  = tex3D(texE, ix-3, iy, iz);
    e[1]  = tex3D(texE, ix-2, iy, iz);
    e[2]  = tex3D(texE, ix-1, iy, iz);
    e[3]  = tex3D(texE, ix,   iy, iz);
    e[4]  = tex3D(texE, ix+1, iy, iz);
    e[5]  = tex3D(texE, ix+2, iy, iz);

    G[0]  = tex3D(texG, ix-3, iy, iz);
    G[1]  = tex3D(texG, ix-2, iy, iz);
    G[2]  = tex3D(texG, ix-1, iy, iz);
    G[3]  = tex3D(texG, ix,   iy, iz);
    G[4]  = tex3D(texG, ix+1, iy, iz);
    G[5]  = tex3D(texG, ix+2, iy, iz);

    P[0]  = tex3D(texP, ix-3, iy, iz);
    P[1]  = tex3D(texP, ix-2, iy, iz);
    P[2]  = tex3D(texP, ix-1, iy, iz);
    P[3]  = tex3D(texP, ix,   iy, iz);
    P[4]  = tex3D(texP, ix+1, iy, iz);
    P[5]  = tex3D(texP, ix+2, iy, iz);

#ifndef NDEBUG
    for (uint_t i = 0; i < 6; ++i)
    {
        assert(r[i] >  0.0f);
        assert(e[i] >  0.0f);
        assert(G[i] >  0.0f);
        assert(P[i] >= 0.0f);
        assert(!isnan(u[i]));
        assert(!isnan(v[i]));
        assert(!isnan(w[i]));
    }
#endif
}


__device__
inline void _load_internal_Y(const uint_t ix, const uint_t iy, const uint_t iz,
        Real * const __restrict__ r,
        Real * const __restrict__ u,
        Real * const __restrict__ v,
        Real * const __restrict__ w,
        Real * const __restrict__ e,
        Real * const __restrict__ G,
        Real * const __restrict__ P,
        const uint_t dummy1,
        const DevicePointer * const dummy2)
{
    // texture only
    r[0]  = tex3D(texR, ix, iy-3, iz);
    r[1]  = tex3D(texR, ix, iy-2, iz);
    r[2]  = tex3D(texR, ix, iy-1, iz);
    r[3]  = tex3D(texR, ix, iy,   iz);
    r[4]  = tex3D(texR, ix, iy+1, iz);
    r[5]  = tex3D(texR, ix, iy+2, iz);

    u[0]  = tex3D(texU, ix, iy-3, iz);
    u[1]  = tex3D(texU, ix, iy-2, iz);
    u[2]  = tex3D(texU, ix, iy-1, iz);
    u[3]  = tex3D(texU, ix, iy,   iz);
    u[4]  = tex3D(texU, ix, iy+1, iz);
    u[5]  = tex3D(texU, ix, iy+2, iz);

    v[0]  = tex3D(texV, ix, iy-3, iz);
    v[1]  = tex3D(texV, ix, iy-2, iz);
    v[2]  = tex3D(texV, ix, iy-1, iz);
    v[3]  = tex3D(texV, ix, iy,   iz);
    v[4]  = tex3D(texV, ix, iy+1, iz);
    v[5]  = tex3D(texV, ix, iy+2, iz);

    w[0]  = tex3D(texW, ix, iy-3, iz);
    w[1]  = tex3D(texW, ix, iy-2, iz);
    w[2]  = tex3D(texW, ix, iy-1, iz);
    w[3]  = tex3D(texW, ix, iy,   iz);
    w[4]  = tex3D(texW, ix, iy+1, iz);
    w[5]  = tex3D(texW, ix, iy+2, iz);

    e[0]  = tex3D(texE, ix, iy-3, iz);
    e[1]  = tex3D(texE, ix, iy-2, iz);
    e[2]  = tex3D(texE, ix, iy-1, iz);
    e[3]  = tex3D(texE, ix, iy,   iz);
    e[4]  = tex3D(texE, ix, iy+1, iz);
    e[5]  = tex3D(texE, ix, iy+2, iz);

    G[0]  = tex3D(texG, ix, iy-3, iz);
    G[1]  = tex3D(texG, ix, iy-2, iz);
    G[2]  = tex3D(texG, ix, iy-1, iz);
    G[3]  = tex3D(texG, ix, iy,   iz);
    G[4]  = tex3D(texG, ix, iy+1, iz);
    G[5]  = tex3D(texG, ix, iy+2, iz);

    P[0]  = tex3D(texP, ix, iy-3, iz);
    P[1]  = tex3D(texP, ix, iy-2, iz);
    P[2]  = tex3D(texP, ix, iy-1, iz);
    P[3]  = tex3D(texP, ix, iy,   iz);
    P[4]  = tex3D(texP, ix, iy+1, iz);
    P[5]  = tex3D(texP, ix, iy+2, iz);

#ifndef NDEBUG
    for (uint_t i = 0; i < 6; ++i)
    {
        assert(r[i] >  0.0f);
        assert(e[i] >  0.0f);
        assert(G[i] >  0.0f);
        assert(P[i] >= 0.0f);
        assert(!isnan(u[i]));
        assert(!isnan(v[i]));
        assert(!isnan(w[i]));
    }
#endif
}


__device__
inline void _load_internal_Z(const uint_t ix, const uint_t iy, const uint_t iz,
        Real * const __restrict__ r,
        Real * const __restrict__ u,
        Real * const __restrict__ v,
        Real * const __restrict__ w,
        Real * const __restrict__ e,
        Real * const __restrict__ G,
        Real * const __restrict__ P,
        const uint_t dummy1,
        const DevicePointer * const dummy2)
{
    // texture only
    r[0]  = tex3D(texR, ix, iy, iz-3);
    r[1]  = tex3D(texR, ix, iy, iz-2);
    r[2]  = tex3D(texR, ix, iy, iz-1);
    r[3]  = tex3D(texR, ix, iy, iz);
    r[4]  = tex3D(texR, ix, iy, iz+1);
    r[5]  = tex3D(texR, ix, iy, iz+2);

    u[0]  = tex3D(texU, ix, iy, iz-3);
    u[1]  = tex3D(texU, ix, iy, iz-2);
    u[2]  = tex3D(texU, ix, iy, iz-1);
    u[3]  = tex3D(texU, ix, iy, iz);
    u[4]  = tex3D(texU, ix, iy, iz+1);
    u[5]  = tex3D(texU, ix, iy, iz+2);

    v[0]  = tex3D(texV, ix, iy, iz-3);
    v[1]  = tex3D(texV, ix, iy, iz-2);
    v[2]  = tex3D(texV, ix, iy, iz-1);
    v[3]  = tex3D(texV, ix, iy, iz);
    v[4]  = tex3D(texV, ix, iy, iz+1);
    v[5]  = tex3D(texV, ix, iy, iz+2);

    w[0]  = tex3D(texW, ix, iy, iz-3);
    w[1]  = tex3D(texW, ix, iy, iz-2);
    w[2]  = tex3D(texW, ix, iy, iz-1);
    w[3]  = tex3D(texW, ix, iy, iz);
    w[4]  = tex3D(texW, ix, iy, iz+1);
    w[5]  = tex3D(texW, ix, iy, iz+2);

    e[0]  = tex3D(texE, ix, iy, iz-3);
    e[1]  = tex3D(texE, ix, iy, iz-2);
    e[2]  = tex3D(texE, ix, iy, iz-1);
    e[3]  = tex3D(texE, ix, iy, iz);
    e[4]  = tex3D(texE, ix, iy, iz+1);
    e[5]  = tex3D(texE, ix, iy, iz+2);

    G[0]  = tex3D(texG, ix, iy, iz-3);
    G[1]  = tex3D(texG, ix, iy, iz-2);
    G[2]  = tex3D(texG, ix, iy, iz-1);
    G[3]  = tex3D(texG, ix, iy, iz);
    G[4]  = tex3D(texG, ix, iy, iz+1);
    G[5]  = tex3D(texG, ix, iy, iz+2);

    P[0]  = tex3D(texP, ix, iy, iz-3);
    P[1]  = tex3D(texP, ix, iy, iz-2);
    P[2]  = tex3D(texP, ix, iy, iz-1);
    P[3]  = tex3D(texP, ix, iy, iz);
    P[4]  = tex3D(texP, ix, iy, iz+1);
    P[5]  = tex3D(texP, ix, iy, iz+2);

#ifndef NDEBUG
    for (uint_t i = 0; i < 6; ++i)
    {
        assert(r[i] >  0.0f);
        assert(e[i] >  0.0f);
        assert(G[i] >  0.0f);
        assert(P[i] >= 0.0f);
        assert(!isnan(u[i]));
        assert(!isnan(v[i]));
        assert(!isnan(w[i]));
    }
#endif
}




template <uint_t ix0, uint_t haloStart, uint_t texStart, uint_t ghostStart>
__device__ inline void _load_1X(const uint_t dummy, const uint_t iy, const uint_t iz,
        Real * const __restrict__ r,
        Real * const __restrict__ u,
        Real * const __restrict__ v,
        Real * const __restrict__ w,
        Real * const __restrict__ e,
        Real * const __restrict__ G,
        Real * const __restrict__ P,
        const uint_t global_iz,
        const DevicePointer * const __restrict__ ghost)
{
    // GMEM
    const uint_t id0 = GHOSTMAPX(ghostStart, iy, iz-3+global_iz);

    r[haloStart] = ghost->r[id0];
    u[haloStart] = ghost->u[id0];
    v[haloStart] = ghost->v[id0];
    w[haloStart] = ghost->w[id0];
    e[haloStart] = ghost->e[id0];
    G[haloStart] = ghost->G[id0];
    P[haloStart] = ghost->P[id0];

    // texture
    r[texStart+0]  = tex3D(texR, ix0+0, iy, iz);
    r[texStart+1]  = tex3D(texR, ix0+1, iy, iz);
    r[texStart+2]  = tex3D(texR, ix0+2, iy, iz);
    r[texStart+3]  = tex3D(texR, ix0+3, iy, iz);
    r[texStart+4]  = tex3D(texR, ix0+4, iy, iz);

    u[texStart+0]  = tex3D(texU, ix0+0, iy, iz);
    u[texStart+1]  = tex3D(texU, ix0+1, iy, iz);
    u[texStart+2]  = tex3D(texU, ix0+2, iy, iz);
    u[texStart+3]  = tex3D(texU, ix0+3, iy, iz);
    u[texStart+4]  = tex3D(texU, ix0+4, iy, iz);

    v[texStart+0]  = tex3D(texV, ix0+0, iy, iz);
    v[texStart+1]  = tex3D(texV, ix0+1, iy, iz);
    v[texStart+2]  = tex3D(texV, ix0+2, iy, iz);
    v[texStart+3]  = tex3D(texV, ix0+3, iy, iz);
    v[texStart+4]  = tex3D(texV, ix0+4, iy, iz);

    w[texStart+0]  = tex3D(texW, ix0+0, iy, iz);
    w[texStart+1]  = tex3D(texW, ix0+1, iy, iz);
    w[texStart+2]  = tex3D(texW, ix0+2, iy, iz);
    w[texStart+3]  = tex3D(texW, ix0+3, iy, iz);
    w[texStart+4]  = tex3D(texW, ix0+4, iy, iz);

    e[texStart+0]  = tex3D(texE, ix0+0, iy, iz);
    e[texStart+1]  = tex3D(texE, ix0+1, iy, iz);
    e[texStart+2]  = tex3D(texE, ix0+2, iy, iz);
    e[texStart+3]  = tex3D(texE, ix0+3, iy, iz);
    e[texStart+4]  = tex3D(texE, ix0+4, iy, iz);

    G[texStart+0]  = tex3D(texG, ix0+0, iy, iz);
    G[texStart+1]  = tex3D(texG, ix0+1, iy, iz);
    G[texStart+2]  = tex3D(texG, ix0+2, iy, iz);
    G[texStart+3]  = tex3D(texG, ix0+3, iy, iz);
    G[texStart+4]  = tex3D(texG, ix0+4, iy, iz);

    P[texStart+0]  = tex3D(texP, ix0+0, iy, iz);
    P[texStart+1]  = tex3D(texP, ix0+1, iy, iz);
    P[texStart+2]  = tex3D(texP, ix0+2, iy, iz);
    P[texStart+3]  = tex3D(texP, ix0+3, iy, iz);
    P[texStart+4]  = tex3D(texP, ix0+4, iy, iz);

#ifndef NDEBUG
    for (uint_t i = 0; i < 6; ++i)
    {
        assert(r[i] >  0.0f);
        assert(e[i] >  0.0f);
        assert(G[i] >  0.0f);
        assert(P[i] >= 0.0f);
        assert(!isnan(u[i]));
        assert(!isnan(v[i]));
        assert(!isnan(w[i]));
    }
#endif
}

template <uint_t iy0, uint_t haloStart, uint_t texStart, uint_t ghostStart>
__device__ inline void _load_1Y(const uint_t ix, const uint_t dummy, const uint_t iz,
        Real * const __restrict__ r,
        Real * const __restrict__ u,
        Real * const __restrict__ v,
        Real * const __restrict__ w,
        Real * const __restrict__ e,
        Real * const __restrict__ G,
        Real * const __restrict__ P,
        const uint_t global_iz,
        const DevicePointer * const __restrict__ ghost)
{
    // GMEM
    const uint_t id0 = GHOSTMAPY(ix, ghostStart, iz-3+global_iz);

    r[haloStart] = ghost->r[id0];
    u[haloStart] = ghost->u[id0];
    v[haloStart] = ghost->v[id0];
    w[haloStart] = ghost->w[id0];
    e[haloStart] = ghost->e[id0];
    G[haloStart] = ghost->G[id0];
    P[haloStart] = ghost->P[id0];

    // texture
    r[texStart+0]  = tex3D(texR, ix, iy0+0, iz);
    r[texStart+1]  = tex3D(texR, ix, iy0+1, iz);
    r[texStart+2]  = tex3D(texR, ix, iy0+2, iz);
    r[texStart+3]  = tex3D(texR, ix, iy0+3, iz);
    r[texStart+4]  = tex3D(texR, ix, iy0+4, iz);

    u[texStart+0]  = tex3D(texU, ix, iy0+0, iz);
    u[texStart+1]  = tex3D(texU, ix, iy0+1, iz);
    u[texStart+2]  = tex3D(texU, ix, iy0+2, iz);
    u[texStart+3]  = tex3D(texU, ix, iy0+3, iz);
    u[texStart+4]  = tex3D(texU, ix, iy0+4, iz);

    v[texStart+0]  = tex3D(texV, ix, iy0+0, iz);
    v[texStart+1]  = tex3D(texV, ix, iy0+1, iz);
    v[texStart+2]  = tex3D(texV, ix, iy0+2, iz);
    v[texStart+3]  = tex3D(texV, ix, iy0+3, iz);
    v[texStart+4]  = tex3D(texV, ix, iy0+4, iz);

    w[texStart+0]  = tex3D(texW, ix, iy0+0, iz);
    w[texStart+1]  = tex3D(texW, ix, iy0+1, iz);
    w[texStart+2]  = tex3D(texW, ix, iy0+2, iz);
    w[texStart+3]  = tex3D(texW, ix, iy0+3, iz);
    w[texStart+4]  = tex3D(texW, ix, iy0+4, iz);

    e[texStart+0]  = tex3D(texE, ix, iy0+0, iz);
    e[texStart+1]  = tex3D(texE, ix, iy0+1, iz);
    e[texStart+2]  = tex3D(texE, ix, iy0+2, iz);
    e[texStart+3]  = tex3D(texE, ix, iy0+3, iz);
    e[texStart+4]  = tex3D(texE, ix, iy0+4, iz);

    G[texStart+0]  = tex3D(texG, ix, iy0+0, iz);
    G[texStart+1]  = tex3D(texG, ix, iy0+1, iz);
    G[texStart+2]  = tex3D(texG, ix, iy0+2, iz);
    G[texStart+3]  = tex3D(texG, ix, iy0+3, iz);
    G[texStart+4]  = tex3D(texG, ix, iy0+4, iz);

    P[texStart+0]  = tex3D(texP, ix, iy0+0, iz);
    P[texStart+1]  = tex3D(texP, ix, iy0+1, iz);
    P[texStart+2]  = tex3D(texP, ix, iy0+2, iz);
    P[texStart+3]  = tex3D(texP, ix, iy0+3, iz);
    P[texStart+4]  = tex3D(texP, ix, iy0+4, iz);

#ifndef NDEBUG
    for (uint_t i = 0; i < 6; ++i)
    {
        assert(r[i] >  0.0f);
        assert(e[i] >  0.0f);
        assert(G[i] >  0.0f);
        assert(P[i] >= 0.0f);
        assert(!isnan(u[i]));
        assert(!isnan(v[i]));
        assert(!isnan(w[i]));
    }
#endif
}


template <uint_t ix0, uint_t haloStart, uint_t texStart, uint_t ghostStart>
__device__ inline void _load_2X(const uint_t dummy, const uint_t iy, const uint_t iz,
        Real * const __restrict__ r,
        Real * const __restrict__ u,
        Real * const __restrict__ v,
        Real * const __restrict__ w,
        Real * const __restrict__ e,
        Real * const __restrict__ G,
        Real * const __restrict__ P,
        const uint_t global_iz,
        const DevicePointer * const __restrict__ ghost)
{
    // GMEM
    const uint_t id0 = GHOSTMAPX(ghostStart+0, iy, iz-3+global_iz);
    const uint_t id1 = GHOSTMAPX(ghostStart+1, iy, iz-3+global_iz);

    r[haloStart+0] = ghost->r[id0];
    r[haloStart+1] = ghost->r[id1];

    u[haloStart+0] = ghost->u[id0];
    u[haloStart+1] = ghost->u[id1];

    v[haloStart+0] = ghost->v[id0];
    v[haloStart+1] = ghost->v[id1];

    w[haloStart+0] = ghost->w[id0];
    w[haloStart+1] = ghost->w[id1];

    e[haloStart+0] = ghost->e[id0];
    e[haloStart+1] = ghost->e[id1];

    G[haloStart+0] = ghost->G[id0];
    G[haloStart+1] = ghost->G[id1];

    P[haloStart+0] = ghost->P[id0];
    P[haloStart+1] = ghost->P[id1];

    // texture
    r[texStart+0]  = tex3D(texR, ix0+0, iy, iz);
    r[texStart+1]  = tex3D(texR, ix0+1, iy, iz);
    r[texStart+2]  = tex3D(texR, ix0+2, iy, iz);
    r[texStart+3]  = tex3D(texR, ix0+3, iy, iz);

    u[texStart+0]  = tex3D(texU, ix0+0, iy, iz);
    u[texStart+1]  = tex3D(texU, ix0+1, iy, iz);
    u[texStart+2]  = tex3D(texU, ix0+2, iy, iz);
    u[texStart+3]  = tex3D(texU, ix0+3, iy, iz);

    v[texStart+0]  = tex3D(texV, ix0+0, iy, iz);
    v[texStart+1]  = tex3D(texV, ix0+1, iy, iz);
    v[texStart+2]  = tex3D(texV, ix0+2, iy, iz);
    v[texStart+3]  = tex3D(texV, ix0+3, iy, iz);

    w[texStart+0]  = tex3D(texW, ix0+0, iy, iz);
    w[texStart+1]  = tex3D(texW, ix0+1, iy, iz);
    w[texStart+2]  = tex3D(texW, ix0+2, iy, iz);
    w[texStart+3]  = tex3D(texW, ix0+3, iy, iz);

    e[texStart+0]  = tex3D(texE, ix0+0, iy, iz);
    e[texStart+1]  = tex3D(texE, ix0+1, iy, iz);
    e[texStart+2]  = tex3D(texE, ix0+2, iy, iz);
    e[texStart+3]  = tex3D(texE, ix0+3, iy, iz);

    G[texStart+0]  = tex3D(texG, ix0+0, iy, iz);
    G[texStart+1]  = tex3D(texG, ix0+1, iy, iz);
    G[texStart+2]  = tex3D(texG, ix0+2, iy, iz);
    G[texStart+3]  = tex3D(texG, ix0+3, iy, iz);

    P[texStart+0]  = tex3D(texP, ix0+0, iy, iz);
    P[texStart+1]  = tex3D(texP, ix0+1, iy, iz);
    P[texStart+2]  = tex3D(texP, ix0+2, iy, iz);
    P[texStart+3]  = tex3D(texP, ix0+3, iy, iz);

#ifndef NDEBUG
    for (uint_t i = 0; i < 6; ++i)
    {
        assert(r[i] >  0.0f);
        assert(e[i] >  0.0f);
        assert(G[i] >  0.0f);
        assert(P[i] >= 0.0f);
        assert(!isnan(u[i]));
        assert(!isnan(v[i]));
        assert(!isnan(w[i]));
    }
#endif
}


template <uint_t iy0, uint_t haloStart, uint_t texStart, uint_t ghostStart>
__device__ inline void _load_2Y(const uint_t ix, const uint_t dummy, const uint_t iz,
        Real * const __restrict__ r,
        Real * const __restrict__ u,
        Real * const __restrict__ v,
        Real * const __restrict__ w,
        Real * const __restrict__ e,
        Real * const __restrict__ G,
        Real * const __restrict__ P,
        const uint_t global_iz,
        const DevicePointer * const __restrict__ ghost)
{
    // GMEM
    const uint_t id0 = GHOSTMAPY(ix, ghostStart+0, iz-3+global_iz);
    const uint_t id1 = GHOSTMAPY(ix, ghostStart+1, iz-3+global_iz);

    r[haloStart+0] = ghost->r[id0];
    r[haloStart+1] = ghost->r[id1];

    u[haloStart+0] = ghost->u[id0];
    u[haloStart+1] = ghost->u[id1];

    v[haloStart+0] = ghost->v[id0];
    v[haloStart+1] = ghost->v[id1];

    w[haloStart+0] = ghost->w[id0];
    w[haloStart+1] = ghost->w[id1];

    e[haloStart+0] = ghost->e[id0];
    e[haloStart+1] = ghost->e[id1];

    G[haloStart+0] = ghost->G[id0];
    G[haloStart+1] = ghost->G[id1];

    P[haloStart+0] = ghost->P[id0];
    P[haloStart+1] = ghost->P[id1];

    // texture
    r[texStart+0]  = tex3D(texR, ix, iy0+0, iz);
    r[texStart+1]  = tex3D(texR, ix, iy0+1, iz);
    r[texStart+2]  = tex3D(texR, ix, iy0+2, iz);
    r[texStart+3]  = tex3D(texR, ix, iy0+3, iz);

    u[texStart+0]  = tex3D(texU, ix, iy0+0, iz);
    u[texStart+1]  = tex3D(texU, ix, iy0+1, iz);
    u[texStart+2]  = tex3D(texU, ix, iy0+2, iz);
    u[texStart+3]  = tex3D(texU, ix, iy0+3, iz);

    v[texStart+0]  = tex3D(texV, ix, iy0+0, iz);
    v[texStart+1]  = tex3D(texV, ix, iy0+1, iz);
    v[texStart+2]  = tex3D(texV, ix, iy0+2, iz);
    v[texStart+3]  = tex3D(texV, ix, iy0+3, iz);

    w[texStart+0]  = tex3D(texW, ix, iy0+0, iz);
    w[texStart+1]  = tex3D(texW, ix, iy0+1, iz);
    w[texStart+2]  = tex3D(texW, ix, iy0+2, iz);
    w[texStart+3]  = tex3D(texW, ix, iy0+3, iz);

    e[texStart+0]  = tex3D(texE, ix, iy0+0, iz);
    e[texStart+1]  = tex3D(texE, ix, iy0+1, iz);
    e[texStart+2]  = tex3D(texE, ix, iy0+2, iz);
    e[texStart+3]  = tex3D(texE, ix, iy0+3, iz);

    G[texStart+0]  = tex3D(texG, ix, iy0+0, iz);
    G[texStart+1]  = tex3D(texG, ix, iy0+1, iz);
    G[texStart+2]  = tex3D(texG, ix, iy0+2, iz);
    G[texStart+3]  = tex3D(texG, ix, iy0+3, iz);

    P[texStart+0]  = tex3D(texP, ix, iy0+0, iz);
    P[texStart+1]  = tex3D(texP, ix, iy0+1, iz);
    P[texStart+2]  = tex3D(texP, ix, iy0+2, iz);
    P[texStart+3]  = tex3D(texP, ix, iy0+3, iz);

#ifndef NDEBUG
    for (uint_t i = 0; i < 6; ++i)
    {
        assert(r[i] >  0.0f);
        assert(e[i] >  0.0f);
        assert(G[i] >  0.0f);
        assert(P[i] >= 0.0f);
        assert(!isnan(u[i]));
        assert(!isnan(v[i]));
        assert(!isnan(w[i]));
    }
#endif
}



template <uint_t ix0, uint_t haloStart, uint_t texStart, uint_t ghostStart>
__device__ inline void _load_3X(const uint_t dummy, const uint_t iy, const uint_t iz,
        Real * const __restrict__ r,
        Real * const __restrict__ u,
        Real * const __restrict__ v,
        Real * const __restrict__ w,
        Real * const __restrict__ e,
        Real * const __restrict__ G,
        Real * const __restrict__ P,
        const uint_t global_iz,
        const DevicePointer * const __restrict__ ghost)
{
    // GMEM
    const uint_t id0 = GHOSTMAPX(ghostStart+0, iy, iz-3+global_iz);
    const uint_t id1 = GHOSTMAPX(ghostStart+1, iy, iz-3+global_iz);
    const uint_t id2 = GHOSTMAPX(ghostStart+2, iy, iz-3+global_iz);

    r[haloStart+0] = ghost->r[id0];
    r[haloStart+1] = ghost->r[id1];
    r[haloStart+2] = ghost->r[id2];

    u[haloStart+0] = ghost->u[id0];
    u[haloStart+1] = ghost->u[id1];
    u[haloStart+2] = ghost->u[id2];

    v[haloStart+0] = ghost->v[id0];
    v[haloStart+1] = ghost->v[id1];
    v[haloStart+2] = ghost->v[id2];

    w[haloStart+0] = ghost->w[id0];
    w[haloStart+1] = ghost->w[id1];
    w[haloStart+2] = ghost->w[id2];

    e[haloStart+0] = ghost->e[id0];
    e[haloStart+1] = ghost->e[id1];
    e[haloStart+2] = ghost->e[id2];

    G[haloStart+0] = ghost->G[id0];
    G[haloStart+1] = ghost->G[id1];
    G[haloStart+2] = ghost->G[id2];

    P[haloStart+0] = ghost->P[id0];
    P[haloStart+1] = ghost->P[id1];
    P[haloStart+2] = ghost->P[id2];

    // texture
    r[texStart+0]  = tex3D(texR, ix0+0, iy, iz);
    r[texStart+1]  = tex3D(texR, ix0+1, iy, iz);
    r[texStart+2]  = tex3D(texR, ix0+2, iy, iz);

    u[texStart+0]  = tex3D(texU, ix0+0, iy, iz);
    u[texStart+1]  = tex3D(texU, ix0+1, iy, iz);
    u[texStart+2]  = tex3D(texU, ix0+2, iy, iz);

    v[texStart+0]  = tex3D(texV, ix0+0, iy, iz);
    v[texStart+1]  = tex3D(texV, ix0+1, iy, iz);
    v[texStart+2]  = tex3D(texV, ix0+2, iy, iz);

    w[texStart+0]  = tex3D(texW, ix0+0, iy, iz);
    w[texStart+1]  = tex3D(texW, ix0+1, iy, iz);
    w[texStart+2]  = tex3D(texW, ix0+2, iy, iz);

    e[texStart+0]  = tex3D(texE, ix0+0, iy, iz);
    e[texStart+1]  = tex3D(texE, ix0+1, iy, iz);
    e[texStart+2]  = tex3D(texE, ix0+2, iy, iz);

    G[texStart+0]  = tex3D(texG, ix0+0, iy, iz);
    G[texStart+1]  = tex3D(texG, ix0+1, iy, iz);
    G[texStart+2]  = tex3D(texG, ix0+2, iy, iz);

    P[texStart+0]  = tex3D(texP, ix0+0, iy, iz);
    P[texStart+1]  = tex3D(texP, ix0+1, iy, iz);
    P[texStart+2]  = tex3D(texP, ix0+2, iy, iz);

#ifndef NDEBUG
    for (uint_t i = 0; i < 6; ++i)
    {
        assert(r[i] >  0.0f);
        assert(e[i] >  0.0f);
        assert(G[i] >  0.0f);
        assert(P[i] >= 0.0f);
        assert(!isnan(u[i]));
        assert(!isnan(v[i]));
        assert(!isnan(w[i]));
    }
#endif
}


template <uint_t iy0, uint_t haloStart, uint_t texStart, uint_t ghostStart>
__device__ inline void _load_3Y(const uint_t ix, const uint_t dummy, const uint_t iz,
        Real * const __restrict__ r,
        Real * const __restrict__ u,
        Real * const __restrict__ v,
        Real * const __restrict__ w,
        Real * const __restrict__ e,
        Real * const __restrict__ G,
        Real * const __restrict__ P,
        const uint_t global_iz,
        const DevicePointer * const __restrict__ ghost)
{
    // GMEM
    const uint_t id0 = GHOSTMAPY(ix, ghostStart+0, iz-3+global_iz);
    const uint_t id1 = GHOSTMAPY(ix, ghostStart+1, iz-3+global_iz);
    const uint_t id2 = GHOSTMAPY(ix, ghostStart+2, iz-3+global_iz);

    r[haloStart+0] = ghost->r[id0];
    r[haloStart+1] = ghost->r[id1];
    r[haloStart+2] = ghost->r[id2];

    u[haloStart+0] = ghost->u[id0];
    u[haloStart+1] = ghost->u[id1];
    u[haloStart+2] = ghost->u[id2];

    v[haloStart+0] = ghost->v[id0];
    v[haloStart+1] = ghost->v[id1];
    v[haloStart+2] = ghost->v[id2];

    w[haloStart+0] = ghost->w[id0];
    w[haloStart+1] = ghost->w[id1];
    w[haloStart+2] = ghost->w[id2];

    e[haloStart+0] = ghost->e[id0];
    e[haloStart+1] = ghost->e[id1];
    e[haloStart+2] = ghost->e[id2];

    G[haloStart+0] = ghost->G[id0];
    G[haloStart+1] = ghost->G[id1];
    G[haloStart+2] = ghost->G[id2];

    P[haloStart+0] = ghost->P[id0];
    P[haloStart+1] = ghost->P[id1];
    P[haloStart+2] = ghost->P[id2];

    // texture
    r[texStart+0]  = tex3D(texR, ix, iy0+0, iz);
    r[texStart+1]  = tex3D(texR, ix, iy0+1, iz);
    r[texStart+2]  = tex3D(texR, ix, iy0+2, iz);

    u[texStart+0]  = tex3D(texU, ix, iy0+0, iz);
    u[texStart+1]  = tex3D(texU, ix, iy0+1, iz);
    u[texStart+2]  = tex3D(texU, ix, iy0+2, iz);

    v[texStart+0]  = tex3D(texV, ix, iy0+0, iz);
    v[texStart+1]  = tex3D(texV, ix, iy0+1, iz);
    v[texStart+2]  = tex3D(texV, ix, iy0+2, iz);

    w[texStart+0]  = tex3D(texW, ix, iy0+0, iz);
    w[texStart+1]  = tex3D(texW, ix, iy0+1, iz);
    w[texStart+2]  = tex3D(texW, ix, iy0+2, iz);

    e[texStart+0]  = tex3D(texE, ix, iy0+0, iz);
    e[texStart+1]  = tex3D(texE, ix, iy0+1, iz);
    e[texStart+2]  = tex3D(texE, ix, iy0+2, iz);

    G[texStart+0]  = tex3D(texG, ix, iy0+0, iz);
    G[texStart+1]  = tex3D(texG, ix, iy0+1, iz);
    G[texStart+2]  = tex3D(texG, ix, iy0+2, iz);

    P[texStart+0]  = tex3D(texP, ix, iy0+0, iz);
    P[texStart+1]  = tex3D(texP, ix, iy0+1, iz);
    P[texStart+2]  = tex3D(texP, ix, iy0+2, iz);

#ifndef NDEBUG
    for (uint_t i = 0; i < 6; ++i)
    {
        assert(r[i] >  0.0f);
        assert(e[i] >  0.0f);
        assert(G[i] >  0.0f);
        assert(P[i] >= 0.0f);
        assert(!isnan(u[i]));
        assert(!isnan(v[i]));
        assert(!isnan(w[i]));
    }
#endif
}


/* __device__ */
/* void _print_stencil(const uint_t bx, const uint_t by, const uint_t tx, const uint_t ty, const Real * const s) */
/* { */
/*     if (bx == blockIdx.x && by == blockIdx.y && tx == threadIdx.x && ty == threadIdx.y) */
/*     { */
/*         printf("Block [%d,%d,%d], Thread [%d,%d,%d], Stencil:\n(",bx,by,blockIdx.z,tx,ty,threadIdx.z); */
/*         for (uint_t i = 0; i < _STENCIL_WIDTH_-1; ++i) */
/*             printf("%f, ", s[i]); */
/*         printf("%f)\n", s[_STENCIL_WIDTH_-1]); */
/*     } */
/* } */




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

struct devPtrSet // 7 fluid quantities
{
    // helper structure to pass compound flow variables as one kernel argument
    Real *r;
    Real *u;
    Real *v;
    Real *w;
    Real *e;
    Real *G;
    Real *P;
    devPtrSet(RealPtrVec_t& c) : r(c[0]), u(c[1]), v(c[2]), w(c[3]), e(c[4]), G(c[5]), P(c[6]) { assert(c.size() == 7); }
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
extern cudaStream_t stream1;
extern cudaStream_t stream2;

// events
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
inline Real _weno_pluss(const Real b, const Real c, const Real d, const Real e, const Real f)
{
#ifndef _WENO3_
    const Real is0 = d*(d*(Real)(10./3.)- e*(Real)(31./3.) + f*(Real)(11./3.)) + e*(e*(Real)(25./3.) - f*(Real)(19./3.)) +    f*f*(Real)(4./3.);
    const Real is1 = c*(c*(Real)(4./3.) - d*(Real)(13./3.) + e*(Real)(5./3.)) + d*(d*(Real)(13./3.)  - e*(Real)(13./3.)) +    e*e*(Real)(4./3.);
    const Real is2 = b*(b*(Real)(4./3.) - c*(Real)(19./3.) + d*(Real)(11./3.)) + c*(c*(Real)(25./3.) - d*(Real)(31./3.)) +    d*d*(Real)(10./3.);

    const Real is0plus = is0 + (Real)WENOEPS;
    const Real is1plus = is1 + (Real)WENOEPS;
    const Real is2plus = is2 + (Real)WENOEPS;

    const Real alpha0 = (Real)(1)*(((Real)1)/(10.0*is0plus*is0plus));
    const Real alpha1 = (Real)(6)*(((Real)1)/(10.0*is1plus*is1plus));
    const Real alpha2 = (Real)(3)*(((Real)1)/(10.0*is2plus*is2plus));
    const Real alphasum = alpha0+alpha1+alpha2;

    const Real omega0=alpha0 * (((Real)1)/alphasum);
    const Real omega1=alpha1 * (((Real)1)/alphasum);
    const Real omega2= 1-omega0-omega1;

    return omega0*((Real)(1./3.)*f-(Real)(7./6.)*e+(Real)(11./6.)*d) + omega1*(-(Real)(1./6.)*e+(Real)(5./6.)*d+(Real)(1./3.)*c) + omega2*((Real)(1./3.)*d+(Real)(5./6.)*c-(Real)(1./6.)*b);

#else

    const Real is0 = (d-e)*(d-e);
    const Real is1 = (d-c)*(d-c);

    const Real alpha0 = (1./3.)/((is0+WENOEPS)*(is0+WENOEPS));
    const Real alpha1 = (2./3.)/((is1+WENOEPS)*(is1+WENOEPS));

    const Real omega0 = alpha0/(alpha0+alpha1);
    const Real omega1 = 1.-omega0;

    return omega0*(1.5*d-.5*e) + omega1*(.5*d+.5*c);
#endif
}


__device__
inline Real _weno_minus(const Real a, const Real b, const Real c, const Real d, const Real e)
{
#ifndef _WENO3_
    const Real is0 = a*(a*(Real)(4./3.)  - b*(Real)(19./3.)  + c*(Real)(11./3.)) + b*(b*(Real)(25./3.)  - c*(Real)(31./3.)) + c*c*(Real)(10./3.);
    const Real is1 = b*(b*(Real)(4./3.)  - c*(Real)(13./3.)  + d*(Real)(5./3.))  + c*(c*(Real)(13./3.)  - d*(Real)(13./3.)) + d*d*(Real)(4./3.);
    const Real is2 = c*(c*(Real)(10./3.) - d*(Real)(31./3.)  + e*(Real)(11./3.)) + d*(d*(Real)(25./3.)  - e*(Real)(19./3.)) + e*e*(Real)(4./3.);

    const Real is0plus = is0 + (Real)WENOEPS;
    const Real is1plus = is1 + (Real)WENOEPS;
    const Real is2plus = is2 + (Real)WENOEPS;

    const Real alpha0 = (Real)(1)*(((Real)1)/(10.0*is0plus*is0plus));
    const Real alpha1 = (Real)(6)*(((Real)1)/(10.0*is1plus*is1plus));
    const Real alpha2 = (Real)(3)*(((Real)1)/(10.0*is2plus*is2plus));
    const Real alphasum = alpha0+alpha1+alpha2;

    const Real omega0=alpha0 * (((Real)1)/alphasum);
    const Real omega1=alpha1 * (((Real)1)/alphasum);
    const Real omega2= 1-omega0-omega1;

    return omega0*((Real)(1.0/3.)*a-(Real)(7./6.)*b+(Real)(11./6.)*c) + omega1*(-(Real)(1./6.)*b+(Real)(5./6.)*c+(Real)(1./3.)*d) + omega2*((Real)(1./3.)*c+(Real)(5./6.)*d-(Real)(1./6.)*e);

#else

    const Real is0 = (c-b)*(c-b);
    const Real is1 = (d-c)*(d-c);

    const Real alpha0 = 1./(3.*(is0+WENOEPS)*(is0+WENOEPS));
    const Real alpha1 = 2./(3.*(is1+WENOEPS)*(is1+WENOEPS));

    const Real omega0=alpha0/(alpha0+alpha1);
    const Real omega1=1.-omega0;

    return omega0*(1.5*c-.5*b) + omega1*(.5*c+.5*d);
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
inline void _print_stencil(const Stencil& s, const uint_t ix, const uint_t iy, const uint_t iz)
{
    printf("(%d,%d,%d) im3 = %f, im2 = %f, im1 = %f, i = %f, ip1 = %f, ip2 = %f\n",ix,iy,iz,s.im3,s.im2,s.im1,s.i,s.ip1,s.ip2);
}


__device__
inline void _load_stencil_tex3D_X(Stencil& s, const texture<float, 3, cudaReadModeElementType> tex, const uint_t ix, const uint_t iy, const uint_t iz)
{
    // read textures (clamped x-addressing)
    s.im3 = tex3D(tex, ix-3, iy, iz);
    s.im2 = tex3D(tex, ix-2, iy, iz);
    s.im1 = tex3D(tex, ix-1, iy, iz);
    s.i   = tex3D(tex, ix,   iy, iz);
    s.ip1 = tex3D(tex, ix+1, iy, iz);
    s.ip2 = tex3D(tex, ix+2, iy, iz);
    assert(!isnan(s.im3));
    assert(!isnan(s.im2));
    assert(!isnan(s.im1));
    assert(!isnan(s.i));
    assert(!isnan(s.ip1));
    assert(!isnan(s.ip2));
}


__device__
inline void _load_3ghosts_X(Real& g0, Real& g1, Real& g2, const Real * const ghosts, const uint_t iy, const uint_t iz)
{
    g0 = ghosts[GHOSTMAPX(0, iy, iz)];
    g1 = ghosts[GHOSTMAPX(1, iy, iz)];
    g2 = ghosts[GHOSTMAPX(2, iy, iz)];
    assert(!isnan(g0));
    assert(!isnan(g1));
    assert(!isnan(g2));
}


__device__
inline void _load_2ghosts_X(Real& g0, Real& g1, const uint_t ix0, const uint_t ix1, const Real * const ghosts, const uint_t iy, const uint_t iz)
{
    assert(ix0 < 3 && ix1 < 3);
    g0 = ghosts[GHOSTMAPX(ix0, iy, iz)];
    g1 = ghosts[GHOSTMAPX(ix1, iy, iz)];
    assert(!isnan(g0));
    assert(!isnan(g1));
}


__device__
inline void _load_1ghost_X(Real& g0, const uint_t ix0, const Real * const ghosts, const uint_t iy, const uint_t iz)
{
    assert(ix0 < 3);
    g0 = ghosts[GHOSTMAPX(ix0, iy, iz)];
    assert(!isnan(g0));
}

__device__
inline void _load_stencil_tex3D_Y(Stencil& s, const texture<float, 3, cudaReadModeElementType> tex, const uint_t ix, const uint_t iy, const uint_t iz)
{
    // read textures (clamped y-addressing)
    s.im3 = tex3D(tex, ix, iy-3, iz);
    s.im2 = tex3D(tex, ix, iy-2, iz);
    s.im1 = tex3D(tex, ix, iy-1, iz);
    s.i   = tex3D(tex, ix, iy,   iz);
    s.ip1 = tex3D(tex, ix, iy+1, iz);
    s.ip2 = tex3D(tex, ix, iy+2, iz);
    assert(!isnan(s.im3));
    assert(!isnan(s.im2));
    assert(!isnan(s.im1));
    assert(!isnan(s.i));
    assert(!isnan(s.ip1));
    assert(!isnan(s.ip2));
}


__device__
inline void _load_3ghosts_Y(Real& g0, Real& g1, Real& g2, const Real * const ghosts, const uint_t ix, const uint_t iz)
{
    g0 = ghosts[GHOSTMAPY(ix, 0, iz)];
    g1 = ghosts[GHOSTMAPY(ix, 1, iz)];
    g2 = ghosts[GHOSTMAPY(ix, 2, iz)];
    assert(!isnan(g0));
    assert(!isnan(g1));
    assert(!isnan(g2));
}


__device__
inline void _load_2ghosts_Y(Real& g0, Real& g1, const uint_t iy0, const uint_t iy1, const Real * const ghosts, const uint_t ix, const uint_t iz)
{
    assert(iy0 < 3 && iy1 < 3);
    g0 = ghosts[GHOSTMAPY(ix, iy0, iz)];
    g1 = ghosts[GHOSTMAPY(ix, iy1, iz)];
    assert(!isnan(g0));
    assert(!isnan(g1));
}


__device__
inline void _load_1ghost_Y(Real& g0, const uint_t iy0, const Real * const ghosts, const uint_t ix, const uint_t iz)
{
    assert(iy0 < 3);
    g0 = ghosts[GHOSTMAPY(ix, iy0, iz)];
    assert(!isnan(g0));
}


__device__
inline void _read_stencil_X(Stencil& s, const texture<float, 3, cudaReadModeElementType> tex,
        const Real * const ghostL, const Real * const ghostR,
        const uint_t ix, const uint_t iy, const uint_t iz, const uint_t global_iz)
{
    switch (ix)
    {
        case 0:
            s.im3 = ghostL[GHOSTMAPX(0, iy, iz-3+global_iz)];
            s.im2 = ghostL[GHOSTMAPX(1, iy, iz-3+global_iz)];
            s.im1 = ghostL[GHOSTMAPX(2, iy, iz-3+global_iz)];
            s.i   = tex3D(tex, ix,   iy, iz);
            s.ip1 = tex3D(tex, ix+1, iy, iz);
            s.ip2 = tex3D(tex, ix+2, iy, iz);
            break;
        case 1:
            s.im3 = ghostL[GHOSTMAPX(1, iy, iz-3+global_iz)];
            s.im2 = ghostL[GHOSTMAPX(2, iy, iz-3+global_iz)];
            s.im1 = tex3D(tex, ix-1, iy, iz);
            s.i   = tex3D(tex, ix,   iy, iz);
            s.ip1 = tex3D(tex, ix+1, iy, iz);
            s.ip2 = tex3D(tex, ix+2, iy, iz);
            break;
        case 2:
            s.im3 = ghostL[GHOSTMAPX(2, iy, iz-3+global_iz)];
            s.im2 = tex3D(tex, ix-2, iy, iz);
            s.im1 = tex3D(tex, ix-1, iy, iz);
            s.i   = tex3D(tex, ix,   iy, iz);
            s.ip1 = tex3D(tex, ix+1, iy, iz);
            s.ip2 = tex3D(tex, ix+2, iy, iz);
            break;
        case NXP1-3:
            s.im3 = tex3D(tex, ix-3, iy, iz);
            s.im2 = tex3D(tex, ix-2, iy, iz);
            s.im1 = tex3D(tex, ix-1, iy, iz);
            s.i   = tex3D(tex, ix,   iy, iz);
            s.ip1 = tex3D(tex, ix+1, iy, iz);
            s.ip2 = ghostR[GHOSTMAPX(0, iy, iz-3+global_iz)];
            break;
        case NXP1-2:
            s.im3 = tex3D(tex, ix-3, iy, iz);
            s.im2 = tex3D(tex, ix-2, iy, iz);
            s.im1 = tex3D(tex, ix-1, iy, iz);
            s.i   = tex3D(tex, ix,   iy, iz);
            s.ip1 = ghostR[GHOSTMAPX(0, iy, iz-3+global_iz)];
            s.ip2 = ghostR[GHOSTMAPX(1, iy, iz-3+global_iz)];
            break;
        case NXP1-1:
            s.im3 = tex3D(tex, ix-3, iy, iz);
            s.im2 = tex3D(tex, ix-2, iy, iz);
            s.im1 = tex3D(tex, ix-1, iy, iz);
            s.i   = ghostR[GHOSTMAPX(0, iy, iz-3+global_iz)];
            s.ip1 = ghostR[GHOSTMAPX(1, iy, iz-3+global_iz)];
            s.ip2 = ghostR[GHOSTMAPX(2, iy, iz-3+global_iz)];
            break;
        default:
            s.im3 = tex3D(tex, ix-3, iy, iz);
            s.im2 = tex3D(tex, ix-2, iy, iz);
            s.im1 = tex3D(tex, ix-1, iy, iz);
            s.i   = tex3D(tex, ix,   iy, iz);
            s.ip1 = tex3D(tex, ix+1, iy, iz);
            s.ip2 = tex3D(tex, ix+2, iy, iz);
            break;
    }
    /* if (ix == 0) */
    /* { */
    /*     s.im3 = ghostL[GHOSTMAPX(0, iy, iz-3+global_iz)]; */
    /*     s.im2 = ghostL[GHOSTMAPX(1, iy, iz-3+global_iz)]; */
    /*     s.im1 = ghostL[GHOSTMAPX(2, iy, iz-3+global_iz)]; */
    /*     s.i   = tex3D(tex, ix,   iy, iz); */
    /*     s.ip1 = tex3D(tex, ix+1, iy, iz); */
    /*     s.ip2 = tex3D(tex, ix+2, iy, iz); */
    /* } */
    /* else if (ix == 1) */
    /* { */
    /*     s.im2 = ghostL[GHOSTMAPX(1, iy, iz-3+global_iz)]; */
    /*     s.im1 = ghostL[GHOSTMAPX(2, iy, iz-3+global_iz)]; */
    /*     s.im1 = tex3D(tex, ix-1, iy, iz); */
    /*     s.i   = tex3D(tex, ix,   iy, iz); */
    /*     s.ip1 = tex3D(tex, ix+1, iy, iz); */
    /*     s.ip2 = tex3D(tex, ix+2, iy, iz); */
    /* } */
    /* else if (ix == 2) */
    /* { */
    /*     s.im1 = ghostL[GHOSTMAPX(2, iy, iz-3+global_iz)]; */
    /*     s.im2 = tex3D(tex, ix-2, iy, iz); */
    /*     s.im1 = tex3D(tex, ix-1, iy, iz); */
    /*     s.i   = tex3D(tex, ix,   iy, iz); */
    /*     s.ip1 = tex3D(tex, ix+1, iy, iz); */
    /*     s.ip2 = tex3D(tex, ix+2, iy, iz); */
    /* } */
    /* else if (ix == NXP1-3) */
    /* { */
    /*     s.im3 = tex3D(tex, ix-3, iy, iz); */
    /*     s.im2 = tex3D(tex, ix-2, iy, iz); */
    /*     s.im1 = tex3D(tex, ix-1, iy, iz); */
    /*     s.i   = tex3D(tex, ix,   iy, iz); */
    /*     s.ip1 = tex3D(tex, ix+1, iy, iz); */
    /*     s.ip2 = ghostR[GHOSTMAPX(0, iy, iz-3+global_iz)]; */
    /* } */
    /* else if (ix == NXP1-2) */
    /* { */
    /*     s.im3 = tex3D(tex, ix-3, iy, iz); */
    /*     s.im2 = tex3D(tex, ix-2, iy, iz); */
    /*     s.im1 = tex3D(tex, ix-1, iy, iz); */
    /*     s.i   = tex3D(tex, ix,   iy, iz); */
    /*     s.ip1 = ghostR[GHOSTMAPX(0, iy, iz-3+global_iz)]; */
    /*     s.ip2 = ghostR[GHOSTMAPX(1, iy, iz-3+global_iz)]; */
    /* } */
    /* else if (ix == NXP1-1) */
    /* { */
    /*     s.im3 = tex3D(tex, ix-3, iy, iz); */
    /*     s.im2 = tex3D(tex, ix-2, iy, iz); */
    /*     s.im1 = tex3D(tex, ix-1, iy, iz); */
    /*     s.i   = ghostR[GHOSTMAPX(0, iy, iz-3+global_iz)]; */
    /*     s.ip1 = ghostR[GHOSTMAPX(1, iy, iz-3+global_iz)]; */
    /*     s.ip2 = ghostR[GHOSTMAPX(2, iy, iz-3+global_iz)]; */
    /* } */
    /* else */
    /* { */
    /*     s.im3 = tex3D(tex, ix-3, iy, iz); */
    /*     s.im2 = tex3D(tex, ix-2, iy, iz); */
    /*     s.im1 = tex3D(tex, ix-1, iy, iz); */
    /*     s.i   = tex3D(tex, ix,   iy, iz); */
    /*     s.ip1 = tex3D(tex, ix+1, iy, iz); */
    /*     s.ip2 = tex3D(tex, ix+2, iy, iz); */
    /* } */
    assert(!isnan(s.im3));
    assert(!isnan(s.im2));
    assert(!isnan(s.im1));
    assert(!isnan(s.i));
    assert(!isnan(s.ip1));
    assert(!isnan(s.ip2));
}


__device__
inline void _read_stencil_Y(Stencil& s, const texture<float, 3, cudaReadModeElementType> tex,
        const Real * const ghostL, const Real * const ghostR,
        const uint_t ix, const uint_t iy, const uint_t iz, const uint_t global_iz)
{
    switch (iy)
    {
        case 0:
            s.im3 = ghostL[GHOSTMAPY(ix, 0, iz-3+global_iz)];
            s.im2 = ghostL[GHOSTMAPY(ix, 1, iz-3+global_iz)];
            s.im1 = ghostL[GHOSTMAPY(ix, 2, iz-3+global_iz)];
            s.i   = tex3D(tex, ix, iy,   iz);
            s.ip1 = tex3D(tex, ix, iy+1, iz);
            s.ip2 = tex3D(tex, ix, iy+2, iz);
            break;
        case 1:
            s.im3 = ghostL[GHOSTMAPY(ix, 1, iz-3+global_iz)];
            s.im2 = ghostL[GHOSTMAPY(ix, 2, iz-3+global_iz)];
            s.im1 = tex3D(tex, ix, iy-1, iz);
            s.i   = tex3D(tex, ix, iy,   iz);
            s.ip1 = tex3D(tex, ix, iy+1, iz);
            s.ip2 = tex3D(tex, ix, iy+2, iz);
            break;
        case 2:
            s.im3 = ghostL[GHOSTMAPY(ix, 2, iz-3+global_iz)];
            s.im2 = tex3D(tex, ix, iy-2, iz);
            s.im1 = tex3D(tex, ix, iy-1, iz);
            s.i   = tex3D(tex, ix, iy,   iz);
            s.ip1 = tex3D(tex, ix, iy+1, iz);
            s.ip2 = tex3D(tex, ix, iy+2, iz);
            break;
        case NYP1-3:
            s.im3 = tex3D(tex, ix, iy-3, iz);
            s.im2 = tex3D(tex, ix, iy-2, iz);
            s.im1 = tex3D(tex, ix, iy-1, iz);
            s.i   = tex3D(tex, ix, iy,   iz);
            s.ip1 = tex3D(tex, ix, iy+1, iz);
            s.ip2 = ghostR[GHOSTMAPY(ix, 0, iz-3+global_iz)];
            break;
        case NYP1-2:
            s.im3 = tex3D(tex, ix, iy-3, iz);
            s.im2 = tex3D(tex, ix, iy-2, iz);
            s.im1 = tex3D(tex, ix, iy-1, iz);
            s.i   = tex3D(tex, ix, iy,   iz);
            s.ip1 = ghostR[GHOSTMAPY(ix, 0, iz-3+global_iz)];
            s.ip2 = ghostR[GHOSTMAPY(ix, 1, iz-3+global_iz)];
            break;
        case NYP1-1:
            s.im3 = tex3D(tex, ix, iy-3, iz);
            s.im2 = tex3D(tex, ix, iy-2, iz);
            s.im1 = tex3D(tex, ix, iy-1, iz);
            s.i   = ghostR[GHOSTMAPY(ix, 0, iz-3+global_iz)];
            s.ip1 = ghostR[GHOSTMAPY(ix, 1, iz-3+global_iz)];
            s.ip2 = ghostR[GHOSTMAPY(ix, 2, iz-3+global_iz)];
            break;
        default:
            s.im3 = tex3D(tex, ix, iy-3, iz);
            s.im2 = tex3D(tex, ix, iy-2, iz);
            s.im1 = tex3D(tex, ix, iy-1, iz);
            s.i   = tex3D(tex, ix, iy,   iz);
            s.ip1 = tex3D(tex, ix, iy+1, iz);
            s.ip2 = tex3D(tex, ix, iy+2, iz);
            break;
    }
    /* if (iy == 0) */
    /* { */
    /*     s.im3 = ghostL[GHOSTMAPY(ix, 0, iz-3+global_iz)]; */
    /*     s.im2 = ghostL[GHOSTMAPY(ix, 1, iz-3+global_iz)]; */
    /*     s.im1 = ghostL[GHOSTMAPY(ix, 2, iz-3+global_iz)]; */
    /*     s.i   = tex3D(tex, ix, iy,   iz); */
    /*     s.ip1 = tex3D(tex, ix, iy+1, iz); */
    /*     s.ip2 = tex3D(tex, ix, iy+2, iz); */
    /* } */
    /* else if (iy == 1) */
    /* { */
    /*     s.im3 = ghostL[GHOSTMAPY(ix, 1, iz-3+global_iz)]; */
    /*     s.im2 = ghostL[GHOSTMAPY(ix, 2, iz-3+global_iz)]; */
    /*     s.im1 = tex3D(tex, ix, iy-1, iz); */
    /*     s.i   = tex3D(tex, ix, iy,   iz); */
    /*     s.ip1 = tex3D(tex, ix, iy+1, iz); */
    /*     s.ip2 = tex3D(tex, ix, iy+2, iz); */
    /* } */
    /* else if (iy == 2) */
    /* { */
    /*     s.im3 = ghostL[GHOSTMAPY(ix, 2, iz-3+global_iz)]; */
    /*     s.im2 = tex3D(tex, ix, iy-2, iz); */
    /*     s.im1 = tex3D(tex, ix, iy-1, iz); */
    /*     s.i   = tex3D(tex, ix, iy,   iz); */
    /*     s.ip1 = tex3D(tex, ix, iy+1, iz); */
    /*     s.ip2 = tex3D(tex, ix, iy+2, iz); */
    /* } */
    /* else if (iy == NYP1-3) */
    /* { */
    /*     s.im3 = tex3D(tex, ix, iy-3, iz); */
    /*     s.im2 = tex3D(tex, ix, iy-2, iz); */
    /*     s.im1 = tex3D(tex, ix, iy-1, iz); */
    /*     s.i   = tex3D(tex, ix, iy,   iz); */
    /*     s.ip1 = tex3D(tex, ix, iy+1, iz); */
    /*     s.ip2 = ghostR[GHOSTMAPY(ix, 0, iz-3+global_iz)]; */
    /* } */
    /* else if (iy == NYP1-2) */
    /* { */
    /*     s.im3 = tex3D(tex, ix, iy-3, iz); */
    /*     s.im2 = tex3D(tex, ix, iy-2, iz); */
    /*     s.im1 = tex3D(tex, ix, iy-1, iz); */
    /*     s.i   = tex3D(tex, ix, iy,   iz); */
    /*     s.ip1 = ghostR[GHOSTMAPY(ix, 0, iz-3+global_iz)]; */
    /*     s.ip2 = ghostR[GHOSTMAPY(ix, 1, iz-3+global_iz)]; */
    /* } */
    /* else if (iy == NYP1-1) */
    /* { */
    /*     s.im3 = tex3D(tex, ix, iy-3, iz); */
    /*     s.im2 = tex3D(tex, ix, iy-2, iz); */
    /*     s.im1 = tex3D(tex, ix, iy-1, iz); */
    /*     s.i   = ghostR[GHOSTMAPY(ix, 0, iz-3+global_iz)]; */
    /*     s.ip1 = ghostR[GHOSTMAPY(ix, 1, iz-3+global_iz)]; */
    /*     s.ip2 = ghostR[GHOSTMAPY(ix, 2, iz-3+global_iz)]; */
    /* } */
    /* else */
    /* { */
    /*     s.im3 = tex3D(tex, ix, iy-3, iz); */
    /*     s.im2 = tex3D(tex, ix, iy-2, iz); */
    /*     s.im1 = tex3D(tex, ix, iy-1, iz); */
    /*     s.i   = tex3D(tex, ix, iy,   iz); */
    /*     s.ip1 = tex3D(tex, ix, iy+1, iz); */
    /*     s.ip2 = tex3D(tex, ix, iy+2, iz); */
    /* } */
    assert(!isnan(s.im3));
    assert(!isnan(s.im2));
    assert(!isnan(s.im1));
    assert(!isnan(s.i));
    assert(!isnan(s.ip1));
    assert(!isnan(s.ip2));
}


__device__
inline void _read_stencil_Z(Stencil& s, const texture<float, 3, cudaReadModeElementType> tex,
        const uint_t ix, const uint_t iy, const uint_t iz)
{
    s.im3 = tex3D(tex, ix, iy, iz-3);
    s.im2 = tex3D(tex, ix, iy, iz-2);
    s.im1 = tex3D(tex, ix, iy, iz-1);
    s.i   = tex3D(tex, ix, iy, iz);
    s.ip1 = tex3D(tex, ix, iy, iz+1);
    s.ip2 = tex3D(tex, ix, iy, iz+2);
    assert(!isnan(s.im3));
    assert(!isnan(s.im2));
    assert(!isnan(s.im1));
    assert(!isnan(s.i));
    assert(!isnan(s.ip1));
    assert(!isnan(s.ip2));
}


__device__
inline void _xfetch_data(const texture<float, 3, cudaReadModeElementType> tex,
        const Real * const ghostL, const Real * const ghostR,
        const uint_t ix, const uint_t iy, const uint_t iz, const uint_t global_iz,
        Real& qm3, Real& qm2, Real& qm1, Real& qp1, Real& qp2, Real& qp3)
{
    // Indexers for the left ghosts.  The starting element is in the
    // outermost layer.  iz must be shifted according to the currently
    // processed chunk, since all of the xghosts reside on the GPU.

    const uint_t idxm1 = GHOSTMAPX(2, iy, iz-3+global_iz);
    const uint_t idxm2 = GHOSTMAPX(1, iy, iz-3+global_iz);
    const uint_t idxm3 = GHOSTMAPX(0, iy, iz-3+global_iz);

    // Indexers for the right ghosts.  The starting element is in the
    // innermost layer.  iz must be shifted according to the currently
    // processed chunk, since all of the xghosts reside on the GPU.

    const uint_t idxp1 = GHOSTMAPX(0, iy, iz-3+global_iz);
    const uint_t idxp2 = GHOSTMAPX(1, iy, iz-3+global_iz);
    const uint_t idxp3 = GHOSTMAPX(2, iy, iz-3+global_iz);

    if (ix == 0)
    {
        qm3 = ghostL[idxm3];
        qm2 = ghostL[idxm2];
        qm1 = ghostL[idxm1];
        qp1 = tex3D(tex, ix,   iy, iz);
        qp2 = tex3D(tex, ix+1, iy, iz);
        qp3 = tex3D(tex, ix+2, iy, iz);
    }
    else if (ix == 1)
    {
        qm3 = ghostL[idxm2];
        qm2 = ghostL[idxm1];
        qm1 = tex3D(tex, ix-1, iy, iz);
        qp1 = tex3D(tex, ix,   iy, iz);
        qp2 = tex3D(tex, ix+1, iy, iz);
        qp3 = tex3D(tex, ix+2, iy, iz);
    }
    else if (ix == 2)
    {
        qm3 = ghostL[idxm1];
        qm2 = tex3D(tex, ix-2, iy, iz);
        qm1 = tex3D(tex, ix-1, iy, iz);
        qp1 = tex3D(tex, ix,   iy, iz);
        qp2 = tex3D(tex, ix+1, iy, iz);
        qp3 = tex3D(tex, ix+2, iy, iz);
    }
    else if (ix == NXP1-3)
    {
        qm3 = tex3D(tex, ix-3, iy, iz);
        qm2 = tex3D(tex, ix-2, iy, iz);
        qm1 = tex3D(tex, ix-1, iy, iz);
        qp1 = tex3D(tex, ix,   iy, iz);
        qp2 = tex3D(tex, ix+1, iy, iz);
        qp3 = ghostR[idxp1];
    }
    else if (ix == NXP1-2)
    {
        qm3 = tex3D(tex, ix-3, iy, iz);
        qm2 = tex3D(tex, ix-2, iy, iz);
        qm1 = tex3D(tex, ix-1, iy, iz);
        qp1 = tex3D(tex, ix,   iy, iz);
        qp2 = ghostR[idxp1];
        qp3 = ghostR[idxp2];
    }
    else if (ix == NXP1-1)
    {
        qm3 = tex3D(tex, ix-3, iy, iz);
        qm2 = tex3D(tex, ix-2, iy, iz);
        qm1 = tex3D(tex, ix-1, iy, iz);
        qp1 = ghostR[idxp1];
        qp2 = ghostR[idxp2];
        qp3 = ghostR[idxp3];
    }
    else
    {
        qm3 = tex3D(tex, ix-3, iy, iz);
        qm2 = tex3D(tex, ix-2, iy, iz);
        qm1 = tex3D(tex, ix-1, iy, iz);
        qp1 = tex3D(tex, ix,   iy, iz);
        qp2 = tex3D(tex, ix+1, iy, iz);
        qp3 = tex3D(tex, ix+2, iy, iz);
    }
    assert(!isnan(qm3));
    assert(!isnan(qm2));
    assert(!isnan(qm1));
    assert(!isnan(qp1));
    assert(!isnan(qp2));
    assert(!isnan(qp3));
}


__device__
inline void _yfetch_data(const texture<float, 3, cudaReadModeElementType> tex,
        const Real* const ghostL, const Real* const ghostR,
        const uint_t ix, const uint_t iy, const uint_t iz, const uint_t global_iz,
        Real& qm3, Real& qm2, Real& qm1, Real& qp1, Real& qp2, Real& qp3)
{
    // Indexers for the left ghosts.  The starting element is in the
    // outermost layer.  iz must be shifted according to the currently
    // processed chunk, since all of the yghosts reside on the GPU.
    const uint_t idxm1 = GHOSTMAPY(ix, 2, iz-3+global_iz);
    const uint_t idxm2 = GHOSTMAPY(ix, 1, iz-3+global_iz);
    const uint_t idxm3 = GHOSTMAPY(ix, 0, iz-3+global_iz);

    // Indexers for the right ghosts.  The starting element is in the
    // innermost layer.  iz must be shifted according to the currently
    // processed chunk, since all of the yghosts reside on the GPU.
    const uint_t idxp1 = GHOSTMAPY(ix, 0, iz-3+global_iz);
    const uint_t idxp2 = GHOSTMAPY(ix, 1, iz-3+global_iz);
    const uint_t idxp3 = GHOSTMAPY(ix, 2, iz-3+global_iz);

    if (iy == 0)
    {
        qm3 = ghostL[idxm3];
        qm2 = ghostL[idxm2];
        qm1 = ghostL[idxm1];
        qp1 = tex3D(tex, ix, iy,   iz);
        qp2 = tex3D(tex, ix, iy+1, iz);
        qp3 = tex3D(tex, ix, iy+2, iz);
    }
    else if (iy == 1)
    {
        qm3 = ghostL[idxm2];
        qm2 = ghostL[idxm1];
        qm1 = tex3D(tex, ix, iy-1, iz);
        qp1 = tex3D(tex, ix, iy,   iz);
        qp2 = tex3D(tex, ix, iy+1, iz);
        qp3 = tex3D(tex, ix, iy+2, iz);
    }
    else if (iy == 2)
    {
        qm3 = ghostL[idxm1];
        qm2 = tex3D(tex, ix, iy-2, iz);
        qm1 = tex3D(tex, ix, iy-1, iz);
        qp1 = tex3D(tex, ix, iy,   iz);
        qp2 = tex3D(tex, ix, iy+1, iz);
        qp3 = tex3D(tex, ix, iy+2, iz);
    }
    else if (iy == NYP1-3)
    {
        qm3 = tex3D(tex, ix, iy-3, iz);
        qm2 = tex3D(tex, ix, iy-2, iz);
        qm1 = tex3D(tex, ix, iy-1, iz);
        qp1 = tex3D(tex, ix, iy,   iz);
        qp2 = tex3D(tex, ix, iy+1, iz);
        qp3 = ghostR[idxp1];
    }
    else if (iy == NYP1-2)
    {
        qm3 = tex3D(tex, ix, iy-3, iz);
        qm2 = tex3D(tex, ix, iy-2, iz);
        qm1 = tex3D(tex, ix, iy-1, iz);
        qp1 = tex3D(tex, ix, iy,   iz);
        qp2 = ghostR[idxp1];
        qp3 = ghostR[idxp2];
    }
    else if (iy == NYP1-1)
    {
        qm3 = tex3D(tex, ix, iy-3, iz);
        qm2 = tex3D(tex, ix, iy-2, iz);
        qm1 = tex3D(tex, ix, iy-1, iz);
        qp1 = ghostR[idxp1];
        qp2 = ghostR[idxp2];
        qp3 = ghostR[idxp3];
    }
    else
    {
        qm3 = tex3D(tex, ix, iy-3, iz);
        qm2 = tex3D(tex, ix, iy-2, iz);
        qm1 = tex3D(tex, ix, iy-1, iz);
        qp1 = tex3D(tex, ix, iy,   iz);
        qp2 = tex3D(tex, ix, iy+1, iz);
        qp3 = tex3D(tex, ix, iy+2, iz);
    }
    assert(!isnan(qm3));
    assert(!isnan(qm2));
    assert(!isnan(qm1));
    assert(!isnan(qp1));
    assert(!isnan(qp2));
    assert(!isnan(qp3));
}


__device__
inline void _zfetch_data(const texture<float, 3, cudaReadModeElementType> tex,
        const uint_t ix, const uint_t iy, const uint_t iz,
        Real& qm3, Real& qm2, Real& qm1, Real& qp1, Real& qp2, Real& qp3)
{
    qm3 = tex3D(tex, ix, iy, iz-3);
    qm2 = tex3D(tex, ix, iy, iz-2);
    qm1 = tex3D(tex, ix, iy, iz-1);
    qp1 = tex3D(tex, ix, iy, iz);
    qp2 = tex3D(tex, ix, iy, iz+1);
    qp3 = tex3D(tex, ix, iy, iz+2);
    assert(!isnan(qm3));
    assert(!isnan(qm2));
    assert(!isnan(qm1));
    assert(!isnan(qp1));
    assert(!isnan(qp2));
    assert(!isnan(qp3));
}


__device__
inline void _char_vel_einfeldt(const Real rm, const Real rp,
        const Real vm, const Real vp,
        const Real pm, const Real pp,
        const Real Gm, const Real Gp,
        const Real Pm, const Real Pp,
        Real& outm, Real& outp) // ? FLOP
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
    assert(rm > 0);
    assert(rp > 0);
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
        const Real sm, const Real sp) // 11 FLOP
{
    const Real facm = rm * (sm - vm);
    const Real facp = rp * (sp - vp);
    return (pp - pm + vm*facm - vp*facp) / (facm - facp);
}


__device__
inline Real _hllc_rho(const Real rm, const Real rp,
        const Real vm, const Real vp,
        const Real sm, const Real sp, const Real ss) // 23 FLOP
{
    /* *
     * The flux computation is split into 4 parts:
     * 1.) Compute signum of s^*, compute s^- and s^+
     * 2.) Compute chi^* and delta of q^* and q
     * 3.) Compute trivial flux
     * 4.) Compute HLLC flux
     * */

    // 1.)
    const int sign_star = ss == 0 ? 0 : (ss < 0 ? -1 : 1);
    const Real s_minus = min(0.0f, sm);
    const Real s_pluss = max(0.0f, sp);

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
    const Real flux = (0.5f*(1 + sign_star)) * (fm + s_minus*q_deltam) + (0.5*(1 - sign_star)) * (fp + s_pluss*q_deltap);
    assert(!isnan(flux));
    return flux;
}


__device__
inline Real _hllc_vel(const Real rm,  const Real rp,
        const Real vm,  const Real vp,
        const Real vdm, const Real vdp,
        const Real sm,  const Real sp,  const Real ss) // 25 FLOP
{
    /* *
     * The flux computation is split into 4 parts:
     * 1.) Compute signum of s^*, compute s^- and s^+
     * 2.) Compute chi^* and delta of q^* and q
     * 3.) Compute trivial flux
     * 4.) Compute HLLC flux
     * */

    // 1.)
    const int sign_star = ss == 0 ? 0 : (ss < 0 ? -1 : 1);
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
    const Real flux = (0.5f*(1 + sign_star)) * (fm + s_minus*q_deltam) + (0.5f*(1 - sign_star)) * (fp + s_pluss*q_deltap);
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
        const Real sm, const Real sp, const Real ss) // 29 FLOP
{
    /* *
     * The flux computation is split into 4 parts:
     * 1.) Compute signum of s^*, compute s^- and s^+
     * 2.) Compute chi^* and delta of q^* and q
     * 3.) Compute trivial flux
     * 4.) Compute HLLC flux
     * */

    // 1.)
    const int sign_star = ss == 0 ? 0 : (ss < 0 ? -1 : 1);
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
    const Real flux = (0.5f*(1 + sign_star)) * (fm + s_minus*q_deltam) + (0.5f*(1 - sign_star)) * (fp + s_pluss*q_deltap);
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
        const Real sm,  const Real sp,  const Real ss) // 59 FLOP
{
    /* *
     * The flux computation is split into 4 parts:
     * 1.) Compute signum of s^*, compute s^- and s^+
     * 2.) Compute chi^* and delta of q^* and q
     * 3.) Compute trivial flux
     * 4.) Compute HLLC flux
     * */

    // 1.)
    const int sign_star = ss == 0 ? 0 : (ss < 0 ? -1 : 1);
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
    const Real flux = (0.5f*(1 + sign_star)) * (fm + s_minus*q_deltam) + (0.5f*(1 - sign_star)) * (fp + s_pluss*q_deltap);
    assert(!isnan(flux));
    return flux;
}


__device__
inline Real _extraterm_hllc_vel(const Real um, const Real up,
        const Real Gm, const Real Gp,
        const Real Pm, const Real Pp,
        const Real sm, const Real sp, const Real ss)
{
    const int sign_star   = (ss == 0) ? 0 : ((ss < 0) ? -1 : 1);
    const Real s_minus   = fminf(0.0f, sm);
    const Real s_pluss   = fmaxf(0.0f, sp);
    const Real chi_starm = (sm - um)/(sm - ss) - 1.0f;
    const Real chi_starp = (sp - up)/(sp - ss) - 1.0f;

    return (0.5f*(1 + sign_star))*(um + s_minus*chi_starm) + (0.5f*(1 - sign_star))*(up + s_pluss*chi_starp);
}


__device__
inline void _fetch_flux(const uint_t ix, const uint_t iy, const uint_t iz,
        const Real * const xflux, const Real * const yflux, const Real * const zflux,
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

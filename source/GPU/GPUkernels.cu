/* *
 * GPUkernels.cu
 *
 * Created by Fabian Wermelinger on 6/25/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <assert.h>
#include <stdio.h>
#include <vector>

#include "GPU.h"
#include "GPUonly.cuh"
#include "CUDA_Timer.cuh"

#define NTHREADS 128

// pointer compound to pass a kernel argument
struct devPtrSet // 7 fluid quantities
{
    Real* r;
    Real* u;
    Real* v;
    Real* w;
    Real* e;
    Real* G;
    Real* P;

    devPtrSet(RealPtrVec_t& c) : r(c[0]), u(c[1]), v(c[2]), w(c[3]), e(c[4]), G(c[5]), P(c[6]) { }
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
extern std::vector<cudaArray_t> d_SOA;

// extraterms for advection equations
extern Real *d_Gm, *d_Gp;
extern Real *d_Pm, *d_Pp;
extern Real *d_hllc_vel;
extern Real *d_sumG, *d_sumP, *d_divU;

// max SOS
extern int* d_maxSOS;

// use non-null stream (async)
extern cudaStream_t stream1;
extern cudaStream_t stream2;

// events
extern cudaEvent_t h2d_tmp_completed;
extern cudaEvent_t divergence_completed;

// texture references
texture<float, 3, cudaReadModeElementType> texR;
texture<float, 3, cudaReadModeElementType> texU;
texture<float, 3, cudaReadModeElementType> texV;
texture<float, 3, cudaReadModeElementType> texW;
texture<float, 3, cudaReadModeElementType> texE;
texture<float, 3, cudaReadModeElementType> texG;
texture<float, 3, cudaReadModeElementType> texP;


///////////////////////////////////////////////////////////////////////////////
//                                  KERNELS                                  //
///////////////////////////////////////////////////////////////////////////////
__global__
void _xextraterm_hllc(const uint_t NX, const uint_t NY, const uint_t NZ,
        const Real * const Gm, const Real * const Gp,
        const Real * const Pm, const Real * const Pp,
        const Real * const vel,
        Real * const sumG, Real * const sumP, Real * const divU)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            const uint_t idx  = ID3(ix, iy, iz, NX, NY); // non-coalesced write!
            const uint_t idxm = ID3(iy, ix, iz, NY, NX+1);
            const uint_t idxp = ID3(iy, ix+1, iz, NY, NX+1);
            sumG[idx] = Gp[idxm] + Gm[idxp];
            sumP[idx] = Pp[idxm] + Pm[idxp];
            divU[idx] = vel[idxp] - vel[idxm];
        }
    }
}


__global__
void _xflux(const uint_t NX, const uint_t NY, const uint_t NZ, const uint_t global_iz,
        devPtrSet ghostL, devPtrSet ghostR, devPtrSet flux,
        Real * const xtra_vel, Real * const xtra_Gm, Real * const xtra_Gp, Real * const xtra_Pm, Real * const xtra_Pp)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        // NOTE: NX = ncells + 1 = number of faces
        //
        // loop over CHUNK_WIDTH slices
        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            /* *
             * 1.) Get cell values
             * 2.) Reconstruct face values (in primitive variables)
             * 3.) Compute characteristic velocities
             * 4.) Compute 7 flux contributions
             * 5.) Compute right hand side for the advection equations
             * */

            ///////////////////////////////////////////////////////////////////
            // 1.) Load data into registers
            ///////////////////////////////////////////////////////////////////
            Real rm3, rm2, rm1, rp1, rp2, rp3;
            _xfetch_data(texR, ghostL.r, ghostR.r, ix, iy, iz, global_iz, NX, NY, rm3, rm2, rm1, rp1, rp2, rp3);
            assert(rm3 > 0); assert(rm2 > 0); assert(rm1 > 0); assert(rp1 > 0); assert(rp2 > 0); assert(rp3 > 0);

            Real um3, um2, um1, up1, up2, up3;
            _xfetch_data(texU, ghostL.u, ghostR.u, ix, iy, iz, global_iz, NX, NY, um3, um2, um1, up1, up2, up3);

            Real vm3, vm2, vm1, vp1, vp2, vp3;
            _xfetch_data(texV, ghostL.v, ghostR.v, ix, iy, iz, global_iz, NX, NY, vm3, vm2, vm1, vp1, vp2, vp3);

            Real wm3, wm2, wm1, wp1, wp2, wp3;
            _xfetch_data(texW, ghostL.w, ghostR.w, ix, iy, iz, global_iz, NX, NY, wm3, wm2, wm1, wp1, wp2, wp3);

            Real em3, em2, em1, ep1, ep2, ep3;
            _xfetch_data(texE, ghostL.e, ghostR.e, ix, iy, iz, global_iz, NX, NY, em3, em2, em1, ep1, ep2, ep3);
            assert(em3 > 0); assert(em2 > 0); assert(em1 > 0); assert(ep1 > 0); assert(ep2 > 0); assert(ep3 > 0);

            Real Gm3, Gm2, Gm1, Gp1, Gp2, Gp3;
            _xfetch_data(texG, ghostL.G, ghostR.G, ix, iy, iz, global_iz, NX, NY, Gm3, Gm2, Gm1, Gp1, Gp2, Gp3);
            assert(Gm3 > 0); assert(Gm2 > 0); assert(Gm1 > 0); assert(Gp1 > 0); assert(Gp2 > 0); assert(Gp3 > 0);

            Real Pm3, Pm2, Pm1, Pp1, Pp2, Pp3;
            _xfetch_data(texP, ghostL.P, ghostR.P, ix, iy, iz, global_iz, NX, NY, Pm3, Pm2, Pm1, Pp1, Pp2, Pp3);
            assert(Pm3 >= 0); assert(Pm2 >= 0); assert(Pm1 >= 0); assert(Pp1 >= 0); assert(Pp2 >= 0); assert(Pp3 >= 0);

            ///////////////////////////////////////////////////////////////////
            // 2.) Reconstruction of primitive values, using WENO5/3
            ///////////////////////////////////////////////////////////////////
            // Reconstruct primitive value p at face f, using WENO5/3
            // rho
            const Real rp = _weno_pluss_clipped(rm2, rm1, rp1, rp2, rp3);
            const Real rm = _weno_minus_clipped(rm3, rm2, rm1, rp1, rp2);
            assert(!isnan(rp)); assert(!isnan(rm));
            // u (convert primitive variable u = (rho*u) / rho)
            um3 /= rm3; um2 /= rm2; um1 /= rm1; up1 /= rp1; up2 /= rp2; up3 /= rp3;
            const Real up = _weno_pluss_clipped(um2, um1, up1, up2, up3);
            const Real um = _weno_minus_clipped(um3, um2, um1, up1, up2);
            assert(!isnan(up)); assert(!isnan(um));
            // v (convert primitive variable v = (rho*v) / rho)
            vm3 /= rm3; vm2 /= rm2; vm1 /= rm1; vp1 /= rp1; vp2 /= rp2; vp3 /= rp3;
            const Real vp = _weno_pluss_clipped(vm2, vm1, vp1, vp2, vp3);
            const Real vm = _weno_minus_clipped(vm3, vm2, vm1, vp1, vp2);
            assert(!isnan(vp)); assert(!isnan(vm));
            // w (convert primitive variable w = (rho*w) / rho)
            wm3 /= rm3; wm2 /= rm2; wm1 /= rm1; wp1 /= rp1; wp2 /= rp2; wp3 /= rp3;
            const Real wp = _weno_pluss_clipped(wm2, wm1, wp1, wp2, wp3);
            const Real wm = _weno_minus_clipped(wm3, wm2, wm1, wp1, wp2);
            assert(!isnan(wp)); assert(!isnan(wm));
            // p (convert primitive variable p = (e - 0.5*rho*(u*u + v*v + w*w) - P) / G
            const Real pm3 = (em3 - 0.5f*rm3*(um3*um3 + vm3*vm3 + wm3*wm3) - Pm3) / Gm3;
            const Real pm2 = (em2 - 0.5f*rm2*(um2*um2 + vm2*vm2 + wm2*wm2) - Pm2) / Gm2;
            const Real pm1 = (em1 - 0.5f*rm1*(um1*um1 + vm1*vm1 + wm1*wm1) - Pm1) / Gm1;
            const Real pp1 = (ep1 - 0.5f*rp1*(up1*up1 + vp1*vp1 + wp1*wp1) - Pp1) / Gp1;
            const Real pp2 = (ep2 - 0.5f*rp2*(up2*up2 + vp2*vp2 + wp2*wp2) - Pp2) / Gp2;
            const Real pp3 = (ep3 - 0.5f*rp3*(up3*up3 + vp3*vp3 + wp3*wp3) - Pp3) / Gp3;
            const Real pp = _weno_pluss_clipped(pm2, pm1, pp1, pp2, pp3);
            const Real pm = _weno_minus_clipped(pm3, pm2, pm1, pp1, pp2);
            assert(!isnan(pp)); assert(!isnan(pm));
            // G
            const Real Gp = _weno_pluss_clipped(Gm2, Gm1, Gp1, Gp2, Gp3);
            const Real Gm = _weno_minus_clipped(Gm3, Gm2, Gm1, Gp1, Gp2);
            assert(!isnan(Gp)); assert(!isnan(Gm));
            // P
            const Real Pp = _weno_pluss_clipped(Pm2, Pm1, Pp1, Pp2, Pp3);
            const Real Pm = _weno_minus_clipped(Pm3, Pm2, Pm1, Pp1, Pp2);
            assert(!isnan(Pp)); assert(!isnan(Pm));

            ///////////////////////////////////////////////////////////////////
            // 3.) Einfeldt characteristic velocities
            ///////////////////////////////////////////////////////////////////
            Real sm, sp;
            _char_vel_einfeldt(rm, rp, um, up, pm, pp, Gm, Gp, Pm, Pp, sm, sp);
            const Real ss = _char_vel_star(rm, rp, um, up, pm, pp, sm, sp);
            assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

            ///////////////////////////////////////////////////////////////////
            // 4.) Compute HLLC fluxes
            ///////////////////////////////////////////////////////////////////
            /* const uint_t idx = ID3(ix, iy, iz, NX, NY); */
            const uint_t idx = ID3(iy, ix, iz, NY, NX);
            const Real fr = _hllc_rho(rm, rp, um, up, sm, sp, ss);
            const Real fu = _hllc_pvel(rm, rp, um, up, pm, pp, sm, sp, ss);
            const Real fv = _hllc_vel(rm, rp, vm, vp, um, up, sm, sp, ss);
            const Real fw = _hllc_vel(rm, rp, wm, wp, um, up, sm, sp, ss);
            const Real fe = _hllc_e(rm, rp, um, up, vm, vp, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss);
            const Real fG = _hllc_rho(Gm, Gp, um, up, sm, sp, ss);
            const Real fP = _hllc_rho(Pm, Pp, um, up, sm, sp, ss);
            assert(!isnan(fr)); assert(!isnan(fu)); assert(!isnan(fv)); assert(!isnan(fw)); assert(!isnan(fe)); assert(!isnan(fG)); assert(!isnan(fP));

            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            ///////////////////////////////////////////////////////////////////
            // 5.)
            ///////////////////////////////////////////////////////////////////
            xtra_vel[idx] = _extraterm_hllc_vel(um, up, Gm, Gp, Pm, Pp, sm, sp, ss);
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
}


__global__
void _yextraterm_hllc(const uint_t NX, const uint_t NY, const uint_t NZ,
        const Real * const Gm, const Real * const Gp,
        const Real * const Pm, const Real * const Pp,
        const Real * const vel,
        Real * const sumG, Real * const sumP, Real * const divU)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            const uint_t idx  = ID3(ix, iy, iz, NX, NY);
            const uint_t idxm = ID3(ix, iy, iz, NX, NY+1);
            const uint_t idxp = ID3(ix, iy+1, iz, NX, NY+1);
            sumG[idx] += Gp[idxm] + Gm[idxp];
            sumP[idx] += Pp[idxm] + Pm[idxp];
            divU[idx] += vel[idxp] - vel[idxm];
        }
    }
}


__global__
void _yflux(const uint_t NX, const uint_t NY, const uint_t NZ, const uint_t global_iz,
        devPtrSet ghostL, devPtrSet ghostR, devPtrSet flux,
        Real * const xtra_vel, Real * const xtra_Gm, Real * const xtra_Gp, Real * const xtra_Pm, Real * const xtra_Pp)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        // NOTE: NY = ncells + 1 = number of faces
        //
        // loop over CHUNK_WIDTH slices
        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            /* *
             * 1.) Get cell values
             * 2.) Reconstruct face values (in primitive variables)
             * 3.) Compute characteristic velocities
             * 4.) Compute 7 flux contributions
             * 5.) Compute right hand side for the advection equations
             * */

            ///////////////////////////////////////////////////////////////////
            // 1.) Load data into registers
            ///////////////////////////////////////////////////////////////////
            Real rm3, rm2, rm1, rp1, rp2, rp3;
            _yfetch_data(texR, ghostL.r, ghostR.r, ix, iy, iz, global_iz, NX, NY, rm3, rm2, rm1, rp1, rp2, rp3);
            assert(rm3 > 0); assert(rm2 > 0); assert(rm1 > 0); assert(rp1 > 0); assert(rp2 > 0); assert(rp3 > 0);

            Real um3, um2, um1, up1, up2, up3;
            _yfetch_data(texU, ghostL.u, ghostR.u, ix, iy, iz, global_iz, NX, NY, um3, um2, um1, up1, up2, up3);

            Real vm3, vm2, vm1, vp1, vp2, vp3;
            _yfetch_data(texV, ghostL.v, ghostR.v, ix, iy, iz, global_iz, NX, NY, vm3, vm2, vm1, vp1, vp2, vp3);

            Real wm3, wm2, wm1, wp1, wp2, wp3;
            _yfetch_data(texW, ghostL.w, ghostR.w, ix, iy, iz, global_iz, NX, NY, wm3, wm2, wm1, wp1, wp2, wp3);

            Real em3, em2, em1, ep1, ep2, ep3;
            _yfetch_data(texE, ghostL.e, ghostR.e, ix, iy, iz, global_iz, NX, NY, em3, em2, em1, ep1, ep2, ep3);
            assert(em3 > 0); assert(em2 > 0); assert(em1 > 0); assert(ep1 > 0); assert(ep2 > 0); assert(ep3 > 0);

            Real Gm3, Gm2, Gm1, Gp1, Gp2, Gp3;
            _yfetch_data(texG, ghostL.G, ghostR.G, ix, iy, iz, global_iz, NX, NY, Gm3, Gm2, Gm1, Gp1, Gp2, Gp3);
            assert(Gm3 > 0); assert(Gm2 > 0); assert(Gm1 > 0); assert(Gp1 > 0); assert(Gp2 > 0); assert(Gp3 > 0);

            Real Pm3, Pm2, Pm1, Pp1, Pp2, Pp3;
            _yfetch_data(texP, ghostL.P, ghostR.P, ix, iy, iz, global_iz, NX, NY, Pm3, Pm2, Pm1, Pp1, Pp2, Pp3);
            assert(Pm3 >= 0); assert(Pm2 >= 0); assert(Pm1 >= 0); assert(Pp1 >= 0); assert(Pp2 >= 0); assert(Pp3 >= 0);

            ///////////////////////////////////////////////////////////////////
            // 2.) Reconstruction of primitive values, using WENO5/3
            ///////////////////////////////////////////////////////////////////
            // rho
            const Real rp = _weno_pluss_clipped(rm2, rm1, rp1, rp2, rp3);
            const Real rm = _weno_minus_clipped(rm3, rm2, rm1, rp1, rp2);
            assert(!isnan(rp)); assert(!isnan(rm));
            // u (convert primitive variable u = (rho*u) / rho)
            um3 /= rm3; um2 /= rm2; um1 /= rm1; up1 /= rp1; up2 /= rp2; up3 /= rp3;
            const Real up = _weno_pluss_clipped(um2, um1, up1, up2, up3);
            const Real um = _weno_minus_clipped(um3, um2, um1, up1, up2);
            assert(!isnan(up)); assert(!isnan(um));
            // v (convert primitive variable v = (rho*v) / rho)
            vm3 /= rm3; vm2 /= rm2; vm1 /= rm1; vp1 /= rp1; vp2 /= rp2; vp3 /= rp3;
            const Real vp = _weno_pluss_clipped(vm2, vm1, vp1, vp2, vp3);
            const Real vm = _weno_minus_clipped(vm3, vm2, vm1, vp1, vp2);
            assert(!isnan(vp)); assert(!isnan(vm));
            // w (convert primitive variable w = (rho*w) / rho)
            wm3 /= rm3; wm2 /= rm2; wm1 /= rm1; wp1 /= rp1; wp2 /= rp2; wp3 /= rp3;
            const Real wp = _weno_pluss_clipped(wm2, wm1, wp1, wp2, wp3);
            const Real wm = _weno_minus_clipped(wm3, wm2, wm1, wp1, wp2);
            assert(!isnan(wp)); assert(!isnan(wm));
            // p (convert primitive variable p = (e - 0.5*rho*(u*u + v*v + w*w) - P) / G
            const Real pm3 = (em3 - 0.5f*rm3*(um3*um3 + vm3*vm3 + wm3*wm3) - Pm3) / Gm3;
            const Real pm2 = (em2 - 0.5f*rm2*(um2*um2 + vm2*vm2 + wm2*wm2) - Pm2) / Gm2;
            const Real pm1 = (em1 - 0.5f*rm1*(um1*um1 + vm1*vm1 + wm1*wm1) - Pm1) / Gm1;
            const Real pp1 = (ep1 - 0.5f*rp1*(up1*up1 + vp1*vp1 + wp1*wp1) - Pp1) / Gp1;
            const Real pp2 = (ep2 - 0.5f*rp2*(up2*up2 + vp2*vp2 + wp2*wp2) - Pp2) / Gp2;
            const Real pp3 = (ep3 - 0.5f*rp3*(up3*up3 + vp3*vp3 + wp3*wp3) - Pp3) / Gp3;
            const Real pp = _weno_pluss_clipped(pm2, pm1, pp1, pp2, pp3);
            const Real pm = _weno_minus_clipped(pm3, pm2, pm1, pp1, pp2);
            assert(!isnan(pp)); assert(!isnan(pm));
            // G
            const Real Gp = _weno_pluss_clipped(Gm2, Gm1, Gp1, Gp2, Gp3);
            const Real Gm = _weno_minus_clipped(Gm3, Gm2, Gm1, Gp1, Gp2);
            assert(!isnan(Gp)); assert(!isnan(Gm));
            // P
            const Real Pp = _weno_pluss_clipped(Pm2, Pm1, Pp1, Pp2, Pp3);
            const Real Pm = _weno_minus_clipped(Pm3, Pm2, Pm1, Pp1, Pp2);
            assert(!isnan(Pp)); assert(!isnan(Pm));

            ///////////////////////////////////////////////////////////////////
            // 3.) Einfeldt characteristic velocities
            ///////////////////////////////////////////////////////////////////
            Real sm, sp;
            _char_vel_einfeldt(rm, rp, vm, vp, pm, pp, Gm, Gp, Pm, Pp, sm, sp);
            const Real ss = _char_vel_star(rm, rp, vm, vp, pm, pp, sm, sp);
            assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

            ///////////////////////////////////////////////////////////////////
            // 4.) Compute HLLC fluxes
            ///////////////////////////////////////////////////////////////////
            const uint_t idx = ID3(ix, iy, iz, NX, NY);
            const Real fr = _hllc_rho(rm, rp, vm, vp, sm, sp, ss);
            const Real fu = _hllc_vel(rm, rp, um, up, vm, vp, sm, sp, ss);
            const Real fv = _hllc_pvel(rm, rp, vm, vp, pm, pp, sm, sp, ss);
            const Real fw = _hllc_vel(rm, rp, wm, wp, vm, vp, sm, sp, ss);
            const Real fe = _hllc_e(rm, rp, vm, vp, um, up, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss);
            const Real fG = _hllc_rho(Gm, Gp, vm, vp, sm, sp, ss);
            const Real fP = _hllc_rho(Pm, Pp, vm, vp, sm, sp, ss);

            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            ///////////////////////////////////////////////////////////////////
            // 5.)
            ///////////////////////////////////////////////////////////////////
            xtra_vel[idx] = _extraterm_hllc_vel(vm, vp, Gm, Gp, Pm, Pp, sm, sp, ss);
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
}


__global__
void _zextraterm_hllc(const uint_t NX, const uint_t NY, const uint_t NZ,
        const Real * const Gm, const Real * const Gp,
        const Real * const Pm, const Real * const Pp,
        const Real * const vel,
        Real * const sumG, Real * const sumP, Real * const divU)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            const uint_t idx  = ID3(ix, iy, iz, NX, NY);
            const uint_t idxm = ID3(ix, iy, iz, NX, NY);
            const uint_t idxp = ID3(ix, iy, iz+1, NX, NY);
            sumG[idx] += Gp[idxm] + Gm[idxp];
            sumP[idx] += Pp[idxm] + Pm[idxp];
            divU[idx] += vel[idxp] - vel[idxm];
        }
    }
}


__global__
void _zflux(const uint_t NX, const uint_t NY, const uint_t NZ,
        devPtrSet flux,
        Real * const xtra_vel, Real * const xtra_Gm, Real * const xtra_Gp, Real * const xtra_Pm, Real * const xtra_Pp)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        // NOTE: NZ = ncells + 1 = number of faces
        //
        // loop over CHUNK_WIDTH+1 slices
        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            /* *
             * 1.) Get cell values
             * 2.) Reconstruct face values (in primitive variables)
             * 3.) Compute characteristic velocities
             * 4.) Compute 7 flux contributions
             * 5.) Compute right hand side for the advection equations
             * */

            ///////////////////////////////////////////////////////////////////
            // 1.) Load data into registers
            ///////////////////////////////////////////////////////////////////
            Real rm3, rm2, rm1, rp1, rp2, rp3;
            _zfetch_data(texR, ix, iy, iz, rm3, rm2, rm1, rp1, rp2, rp3);
            assert(rm3 > 0); assert(rm2 > 0); assert(rm1 > 0); assert(rp1 > 0); assert(rp2 > 0); assert(rp3 > 0);

            Real um3, um2, um1, up1, up2, up3;
            _zfetch_data(texU, ix, iy, iz, um3, um2, um1, up1, up2, up3);

            Real vm3, vm2, vm1, vp1, vp2, vp3;
            _zfetch_data(texV, ix, iy, iz, vm3, vm2, vm1, vp1, vp2, vp3);

            Real wm3, wm2, wm1, wp1, wp2, wp3;
            _zfetch_data(texW, ix, iy, iz, wm3, wm2, wm1, wp1, wp2, wp3);

            Real em3, em2, em1, ep1, ep2, ep3;
            _zfetch_data(texE, ix, iy, iz, em3, em2, em1, ep1, ep2, ep3);
            assert(em3 > 0); assert(em2 > 0); assert(em1 > 0); assert(ep1 > 0); assert(ep2 > 0); assert(ep3 > 0);

            Real Gm3, Gm2, Gm1, Gp1, Gp2, Gp3;
            _zfetch_data(texG, ix, iy, iz, Gm3, Gm2, Gm1, Gp1, Gp2, Gp3);
            assert(Gm3 > 0); assert(Gm2 > 0); assert(Gm1 > 0); assert(Gp1 > 0); assert(Gp2 > 0); assert(Gp3 > 0);

            Real Pm3, Pm2, Pm1, Pp1, Pp2, Pp3;
            _zfetch_data(texP, ix, iy, iz, Pm3, Pm2, Pm1, Pp1, Pp2, Pp3);
            assert(Pm3 >= 0); assert(Pm2 >= 0); assert(Pm1 >= 0); assert(Pp1 >= 0); assert(Pp2 >= 0); assert(Pp3 >= 0);

            ///////////////////////////////////////////////////////////////////
            // 2.) Reconstruction of primitive values, using WENO5/3
            ///////////////////////////////////////////////////////////////////
            // rho
            const Real rp = _weno_pluss_clipped(rm2, rm1, rp1, rp2, rp3);
            const Real rm = _weno_minus_clipped(rm3, rm2, rm1, rp1, rp2);
            assert(!isnan(rp)); assert(!isnan(rm));
            // u (convert primitive variable u = (rho*u) / rho)
            um3 /= rm3; um2 /= rm2; um1 /= rm1; up1 /= rp1; up2 /= rp2; up3 /= rp3;
            const Real up = _weno_pluss_clipped(um2, um1, up1, up2, up3);
            const Real um = _weno_minus_clipped(um3, um2, um1, up1, up2);
            assert(!isnan(up)); assert(!isnan(um));
            // v (convert primitive variable v = (rho*v) / rho)
            vm3 /= rm3; vm2 /= rm2; vm1 /= rm1; vp1 /= rp1; vp2 /= rp2; vp3 /= rp3;
            const Real vp = _weno_pluss_clipped(vm2, vm1, vp1, vp2, vp3);
            const Real vm = _weno_minus_clipped(vm3, vm2, vm1, vp1, vp2);
            assert(!isnan(vp)); assert(!isnan(vm));
            // w (convert primitive variable w = (rho*w) / rho)
            wm3 /= rm3; wm2 /= rm2; wm1 /= rm1; wp1 /= rp1; wp2 /= rp2; wp3 /= rp3;
            const Real wp = _weno_pluss_clipped(wm2, wm1, wp1, wp2, wp3);
            const Real wm = _weno_minus_clipped(wm3, wm2, wm1, wp1, wp2);
            assert(!isnan(wp)); assert(!isnan(wm));
            // p (convert primitive variable p = (e - 0.5*rho*(u*u + v*v + w*w) - P) / G
            const Real pm3 = (em3 - 0.5f*rm3*(um3*um3 + vm3*vm3 + wm3*wm3) - Pm3) / Gm3;
            const Real pm2 = (em2 - 0.5f*rm2*(um2*um2 + vm2*vm2 + wm2*wm2) - Pm2) / Gm2;
            const Real pm1 = (em1 - 0.5f*rm1*(um1*um1 + vm1*vm1 + wm1*wm1) - Pm1) / Gm1;
            const Real pp1 = (ep1 - 0.5f*rp1*(up1*up1 + vp1*vp1 + wp1*wp1) - Pp1) / Gp1;
            const Real pp2 = (ep2 - 0.5f*rp2*(up2*up2 + vp2*vp2 + wp2*wp2) - Pp2) / Gp2;
            const Real pp3 = (ep3 - 0.5f*rp3*(up3*up3 + vp3*vp3 + wp3*wp3) - Pp3) / Gp3;
            const Real pp = _weno_pluss_clipped(pm2, pm1, pp1, pp2, pp3);
            const Real pm = _weno_minus_clipped(pm3, pm2, pm1, pp1, pp2);
            assert(!isnan(pp)); assert(!isnan(pm));
            // G
            const Real Gp = _weno_pluss_clipped(Gm2, Gm1, Gp1, Gp2, Gp3);
            const Real Gm = _weno_minus_clipped(Gm3, Gm2, Gm1, Gp1, Gp2);
            assert(!isnan(Gp)); assert(!isnan(Gm));
            // P
            const Real Pp = _weno_pluss_clipped(Pm2, Pm1, Pp1, Pp2, Pp3);
            const Real Pm = _weno_minus_clipped(Pm3, Pm2, Pm1, Pp1, Pp2);
            assert(!isnan(Pp)); assert(!isnan(Pm));

            ///////////////////////////////////////////////////////////////////
            // 3.) Einfeldt characteristic velocities
            ///////////////////////////////////////////////////////////////////
            Real sm, sp;
            _char_vel_einfeldt(rm, rp, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp);
            const Real ss = _char_vel_star(rm, rp, wm, wp, pm, pp, sm, sp);
            assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

            ///////////////////////////////////////////////////////////////////
            // 4.) Compute HLLC fluxes
            ///////////////////////////////////////////////////////////////////
            const uint_t idx = ID3(ix, iy, iz, NX, NY);
            const Real fr = _hllc_rho(rm, rp, wm, wp, sm, sp, ss);
            const Real fu = _hllc_vel(rm, rp, um, up, wm, wp, sm, sp, ss);
            const Real fv = _hllc_vel(rm, rp, vm, vp, wm, wp, sm, sp, ss);
            const Real fw = _hllc_pvel(rm, rp, wm, wp, pm, pp, sm, sp, ss);
            const Real fe = _hllc_e(rm, rp, wm, wp, um, up, vm, vp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss);
            const Real fG = _hllc_rho(Gm, Gp, wm, wp, sm, sp, ss);
            const Real fP = _hllc_rho(Pm, Pp, wm, wp, sm, sp, ss);

            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            ///////////////////////////////////////////////////////////////////
            // 5.)
            ///////////////////////////////////////////////////////////////////
            xtra_vel[idx] = _extraterm_hllc_vel(wm, wp, Gm, Gp, Pm, Pp, sm, sp, ss);
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
}


__global__
void _divergence(const uint_t NX, const uint_t NY, const uint_t NZ,
        const devPtrSet xflux, const devPtrSet yflux, const devPtrSet zflux,
        devPtrSet rhs, const Real a, const Real dtinvh, const devPtrSet tmp,
        const Real * const sumG, const Real * const sumP, const Real * const divU)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        Real fxp, fxm, fyp, fym, fzp, fzm;
        Real rhs_reg;
        const Real factor6 = 1.0f / 6.0f;
        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            const uint_t idx = ID3(ix, iy, iz, NX, NY);

            _fetch_flux(ix, iy, iz, NX, NY, xflux.r, yflux.r, zflux.r, fxp, fxm, fyp, fym, fzp, fzm);
            rhs_reg = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.r[idx] = a*tmp.r[idx] - rhs_reg;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.u, yflux.u, zflux.u, fxp, fxm, fyp, fym, fzp, fzm);
            rhs_reg = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.u[idx] = a*tmp.u[idx] - rhs_reg;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.v, yflux.v, zflux.v, fxp, fxm, fyp, fym, fzp, fzm);
            rhs_reg = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.v[idx] = a*tmp.v[idx] - rhs_reg;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.w, yflux.w, zflux.w, fxp, fxm, fyp, fym, fzp, fzm);
            rhs_reg = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.w[idx] = a*tmp.w[idx] - rhs_reg;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.e, yflux.e, zflux.e, fxp, fxm, fyp, fym, fzp, fzm);
            rhs_reg = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.e[idx] = a*tmp.e[idx] - rhs_reg;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.G, yflux.G, zflux.G, fxp, fxm, fyp, fym, fzp, fzm);
            rhs_reg = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm   - divU[idx] * sumG[idx] * factor6);
            rhs.G[idx] = a*tmp.G[idx] - rhs_reg;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.P, yflux.P, zflux.P, fxp, fxm, fyp, fym, fzp, fzm);
            rhs_reg = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm   - divU[idx] * sumP[idx] * factor6);
            rhs.P[idx] = a*tmp.P[idx] - rhs_reg;
        }
    }
}


__global__
void _update(const uint_t NX, const uint_t NY, const uint_t NZ,
        const Real b, devPtrSet tmp, const devPtrSet rhs)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            const uint_t idx = ID3(ix, iy, iz, NX, NY);

            const Real r = tex3D(texR, ix, iy, iz+3);
            const Real u = tex3D(texU, ix, iy, iz+3);
            const Real v = tex3D(texV, ix, iy, iz+3);
            const Real w = tex3D(texW, ix, iy, iz+3);
            const Real e = tex3D(texE, ix, iy, iz+3);
            const Real G = tex3D(texG, ix, iy, iz+3);
            const Real P = tex3D(texP, ix, iy, iz+3);

            // this overwrites the rhs from the previous stage, stored in tmp,
            // with the updated solution.
            tmp.r[idx] = b*rhs.r[idx] + r;
            tmp.u[idx] = b*rhs.u[idx] + u;
            tmp.v[idx] = b*rhs.v[idx] + v;
            tmp.w[idx] = b*rhs.w[idx] + w;
            tmp.e[idx] = b*rhs.e[idx] + e;
            tmp.G[idx] = b*rhs.G[idx] + G;
            tmp.P[idx] = b*rhs.P[idx] + P;
        }
    }
}


__global__
void _maxSOS(const uint_t NX, const uint_t NY, const uint_t NZ, int* g_maxSOS)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    const uint_t loc_idx = blockDim.x * threadIdx.y + threadIdx.x;
    __shared__ Real block_sos[NTHREADS];
    block_sos[loc_idx] = 0.0f;

    if (ix < NX && iy < NY)
    {
        Real sos = 0.0f;

        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            const Real r = tex3D(texR, ix, iy, iz);
            const Real u = tex3D(texU, ix, iy, iz);
            const Real v = tex3D(texV, ix, iy, iz);
            const Real w = tex3D(texW, ix, iy, iz);
            const Real e = tex3D(texE, ix, iy, iz);
            const Real G = tex3D(texG, ix, iy, iz);
            const Real P = tex3D(texP, ix, iy, iz);

            const Real p = (e - 0.5f*(u*u + v*v + w*w)/r - P) / G;
            const Real c = sqrtf(((p + P) / G + p) / r);

            sos = fmaxf(sos, c + fmaxf(fmaxf(fabsf(u), fabsf(v)), fabsf(w)) / r);
        }
        block_sos[loc_idx] = sos;
        __syncthreads();

        if (0 == loc_idx)
        {
            for (int i = 1; i < NTHREADS; ++i)
                sos = fmaxf(sos, block_sos[i]);
            atomicMax(g_maxSOS, __float_as_int(sos));
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
//                              KERNEL WRAPPERS                              //
///////////////////////////////////////////////////////////////////////////////
extern "C"
{
    void GPU::xflux(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH, const uint_t global_iz)
    {
        devPtrSet xghostL(d_xgl);
        devPtrSet xghostR(d_xgr);
        devPtrSet xflux(d_xflux);

        {
            const uint_t XSize = BSX_GPU + 1;
            const dim3 grid(XSize, (BSY_GPU + NTHREADS -1) / NTHREADS, 1);
            const dim3 blocks(1, NTHREADS, 1);

            /* const uint_t XSize = BSX_GPU + 1; */
            /* const dim3 grid(XSize, 1, 1); */
            /* const dim3 blocks(1, 256, 1); */

            GPUtimer kernel;
            kernel.start(stream1);
            _xflux<<<grid, blocks, 0, stream1>>>(XSize, BSY_GPU, CHUNK_WIDTH, global_iz, xghostL, xghostR, xflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            kernel.stop(stream1);
            kernel.print("[_xflux Kernel]: ");
        }

        {
            const uint_t XSize = BSX_GPU;
            const dim3 grid(XSize, (BSY_GPU + NTHREADS -1) / NTHREADS, 1);
            const dim3 blocks(1, NTHREADS, 1);
            GPUtimer xextra;
            xextra.start(stream1);
            _xextraterm_hllc<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            xextra.stop(stream1);
            xextra.print("[_xextraterm Kernel]: ");
        }
    }


    void GPU::yflux(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH, const uint_t global_iz)
    {
        devPtrSet yghostL(d_ygl);
        devPtrSet yghostR(d_ygr);
        devPtrSet yflux(d_yflux);

        {
            const uint_t YSize = BSY_GPU + 1;
            const dim3 grid((BSX_GPU + NTHREADS -1) / NTHREADS, YSize, 1);
            const dim3 blocks(NTHREADS, 1, 1);

            GPUtimer kernel;
            kernel.start(stream1);
            _yflux<<<grid, blocks, 0, stream1>>>(BSX_GPU, YSize, CHUNK_WIDTH, global_iz, yghostL, yghostR, yflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            kernel.stop(stream1);
            kernel.print("[_yflux Kernel]: ");
        }

        {
            const uint_t YSize = BSY_GPU;
            const dim3 grid((BSX_GPU + NTHREADS -1) / NTHREADS, YSize, 1);
            const dim3 blocks(NTHREADS, 1, 1);

            GPUtimer yextra;
            yextra.start(stream1);
            _yextraterm_hllc<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            yextra.stop(stream1);
            yextra.print("[_yextraterm Kernel]: ");
        }
    }


    void GPU::zflux(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH)
    {
        devPtrSet zflux(d_zflux);

        {
            const uint_t ZSize = CHUNK_WIDTH + 1;
            const dim3 grid((BSX_GPU + NTHREADS -1) / NTHREADS, BSY_GPU, 1);
            const dim3 blocks(NTHREADS, 1, 1);

            GPUtimer kernel;
            kernel.start(stream1);
            _zflux<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, ZSize, zflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            kernel.stop(stream1);
            kernel.print("[_zflux Kernel]: ");
        }

        {
            const dim3 grid((BSX_GPU + NTHREADS -1) / NTHREADS, BSY_GPU, 1);
            const dim3 blocks(NTHREADS, 1, 1);

            GPUtimer zextra;
            zextra.start(stream1);
            _zextraterm_hllc<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            zextra.stop(stream1);
            zextra.print("[_zextraterm Kernel]: ");
        }
    }


    void GPU::divergence(const Real a, const Real dtinvh, const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH)
    {
        cudaStreamWaitEvent(stream1, h2d_tmp_completed, 0);

        devPtrSet xflux(d_xflux);
        devPtrSet yflux(d_yflux);
        devPtrSet zflux(d_zflux);
        devPtrSet rhs(d_rhs);
        devPtrSet tmp(d_tmp);

        const dim3 grid((BSX_GPU + NTHREADS -1) / NTHREADS, BSY_GPU, 1);
        const dim3 blocks(NTHREADS, 1, 1);

        GPUtimer kernel;
        kernel.start(stream1);
        _divergence<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, xflux, yflux, zflux, rhs, a, dtinvh, tmp, d_sumG, d_sumP, d_divU);
        kernel.stop(stream1);
        kernel.print("[_divergence Kernel]: ");

        cudaEventRecord(divergence_completed, stream1);
    }


    void GPU::update(const Real b, const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH)
    {
        devPtrSet tmp(d_tmp);
        devPtrSet rhs(d_rhs);

        const dim3 grid((BSX_GPU + NTHREADS -1) / NTHREADS, BSY_GPU, 1);
        const dim3 blocks(NTHREADS, 1, 1);

        GPUtimer kernel;
        kernel.start(stream1);
        _update<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, b, tmp, rhs);
        kernel.stop(stream1);
        kernel.print("[_update Kernel]: ");
    }


    void GPU::MaxSpeedOfSound(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH)
    {
        const dim3 grid((BSX_GPU + NTHREADS -1) / NTHREADS, BSY_GPU, 1);
        const dim3 blocks(NTHREADS, 1, 1);

        GPUtimer kernel;
        kernel.start(stream1);
        _maxSOS<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, d_maxSOS);
        kernel.stop(stream1);
        kernel.print("[_maxSOS Kernel]: ");
    }
}


///////////////////////////////////////////////////////////////////////////////
//                                   UTILS                                   //
///////////////////////////////////////////////////////////////////////////////
static void _bindTexture(texture<float, 3, cudaReadModeElementType> tex, cudaArray_t d_ptr)
{
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<Real>();
    tex.addressMode[0]        = cudaAddressModeClamp;
    tex.addressMode[1]        = cudaAddressModeClamp;
    tex.addressMode[2]        = cudaAddressModeClamp;
    tex.channelDesc           = fmt;
    tex.filterMode            = cudaFilterModePoint;
    tex.mipmapFilterMode      = cudaFilterModePoint;
    tex.normalized            = false;

    cudaBindTextureToArray(&tex, d_ptr, &fmt);
}


extern "C"
{
    void GPU::bind_textures()
    {
        _bindTexture(texR, d_SOA[0]);
        _bindTexture(texU, d_SOA[1]);
        _bindTexture(texV, d_SOA[2]);
        _bindTexture(texW, d_SOA[3]);
        _bindTexture(texE, d_SOA[4]);
        _bindTexture(texG, d_SOA[5]);
        _bindTexture(texP, d_SOA[6]);
    }


    void GPU::unbind_textures()
    {
        cudaUnbindTexture(&texR);
        cudaUnbindTexture(&texU);
        cudaUnbindTexture(&texV);
        cudaUnbindTexture(&texW);
        cudaUnbindTexture(&texE);
        cudaUnbindTexture(&texG);
        cudaUnbindTexture(&texP);
    }
}

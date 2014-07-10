/* *
 * GPUkernels.cu
 *
 * Created by Fabian Wermelinger on 6/25/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <assert.h>
#include <stdio.h>
#include <vector>

#include "GPU.h" // includes Types.h & wrapper declarations
#include "GPUonly.cuh"
#include "CUDA_Timer.cuh"


///////////////////////////////////////////////////////////////////////////////
//                             DEVICE FUNCTIONS                              //
///////////////////////////////////////////////////////////////////////////////
__device__
static void _compute_flux_all(devPtrSet& flux, Stencil& r, Stencil& u, Stencil& v, Stencil& w, Stencil& e, Stencil& G, Stencil& P, const uint_t flux_id,
        Real * const xtra_vel, Real * const xtra_Gm, Real * const xtra_Gp, Real * const xtra_Pm, Real * const xtra_Pp)
{
    /* *
     * 1.) Reconstruct primitive values using WENO5/WENO3
     * 2.) Compute characteristic velocities
     * 3.) Compute fluxes
     * 4.) Compute RHS for advection of G and P
     * */

    // cont here!
    // 1.)
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

    /*         /////////////////////////////////////////////////////////////////// */
    /*         // 3.) Einfeldt characteristic velocities */
    /*         /////////////////////////////////////////////////////////////////// */
    /*         Real sm, sp; */
    /*         _char_vel_einfeldt(rm, rp, um, up, pm, pp, Gm, Gp, Pm, Pp, sm, sp); */
    /*         const Real ss = _char_vel_star(rm, rp, um, up, pm, pp, sm, sp); */
    /*         assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss)); */

    /*         /////////////////////////////////////////////////////////////////// */
    /*         // 4.) Compute HLLC fluxes */
    /*         /////////////////////////////////////////////////////////////////// */
    /*         const Real fr = _hllc_rho(rm, rp, um, up, sm, sp, ss); */
    /*         const Real fu = _hllc_pvel(rm, rp, um, up, pm, pp, sm, sp, ss); */
    /*         const Real fv = _hllc_vel(rm, rp, vm, vp, um, up, sm, sp, ss); */
    /*         const Real fw = _hllc_vel(rm, rp, wm, wp, um, up, sm, sp, ss); */
    /*         const Real fe = _hllc_e(rm, rp, um, up, vm, vp, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss); */
    /*         const Real fG = _hllc_rho(Gm, Gp, um, up, sm, sp, ss); */
    /*         const Real fP = _hllc_rho(Pm, Pp, um, up, sm, sp, ss); */
    /*         assert(!isnan(fr)); assert(!isnan(fu)); assert(!isnan(fv)); assert(!isnan(fw)); assert(!isnan(fe)); assert(!isnan(fG)); assert(!isnan(fP)); */

    /*         const uint_t idx = ID3(iy, ix, iz, NY, NXp1); */
    /*         flux.r[idx] = fr; */
    /*         flux.u[idx] = fu; */
    /*         flux.v[idx] = fv; */
    /*         flux.w[idx] = fw; */
    /*         flux.e[idx] = fe; */
    /*         flux.G[idx] = fG; */
    /*         flux.P[idx] = fP; */

    /*         /////////////////////////////////////////////////////////////////// */
    /*         // 5.) RHS for advection equations */
    /*         /////////////////////////////////////////////////////////////////// */
    /*         xtra_vel[idx] = _extraterm_hllc_vel(um, up, Gm, Gp, Pm, Pp, sm, sp, ss); */
    /*         xtra_Gm[idx]  = Gm; */
    /*         xtra_Gp[idx]  = Gp; */
    /*         xtra_Pm[idx]  = Pm; */
    /*         xtra_Pp[idx]  = Pp; */
}


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
    /* *
     * Computes x-contribution for the right hand side of the advection
     * equations.  Maps two values on cell faces to one value at the cell
     * center.  NOTE: The assignment here is "="
     * */
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            const uint_t idx  = ID3(iy, ix,   iz, NY, NX);
            const uint_t idxm = ID3(iy, ix,   iz, NY, NX+1);
            const uint_t idxp = ID3(iy, ix+1, iz, NY, NX+1);
            sumG[idx] = Gp[idxm]  + Gm[idxp];
            sumP[idx] = Pp[idxm]  + Pm[idxp];
            divU[idx] = vel[idxp] - vel[idxm];
        }
    }
}


__global__
void _xflux(const uint_t NXp1, const uint_t NY, const uint_t NZ, const uint_t global_iz,
        devPtrSet ghostL, devPtrSet ghostR, devPtrSet flux,
        Real * const xtra_vel, Real * const xtra_Gm, Real * const xtra_Gp, Real * const xtra_Pm, Real * const xtra_Pp)
{
    /* *
     * Notes:
     * ======
     * 1.) NXp1 = NX + 1
     * 2.) NX = NodeBlock::sizeX
     * 3.) NY = NodeBlock::sizeY
     * 4.) NZ = number of slices for currently processed chunk
     * 5.) global_iz is the iz-coordinate in index space of the NodeBlock for
     *     the first slice of the currently processed chunk.  It is needed if
     *     all of the x-/yghosts are uploaded to the GPU prior to processing
     *     the chunks sequentially.  Currently global_iz = 0, since x-/yghosts
     *     are uploaded per chunk.
     * */
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    /* *
     * The following code requires 1D thread blocks with dimension
     * dim3(1, _NTHREADS_, 1)
     * on a 2D grid with dimension
     * dim3(NXp1, (NY + _NTHREADS_ - 1) / _NTHREADS_, 1)
     *
     * To account for the irregularity of the ghosts, the execution is split up
     * into 3 zones:
     * 1.) left ghosts (first 3 if)
     * 2.) right ghosts (next 3 if)
     * 3.) interior (last if)
     * Due to the thread block specification, there will be no warp divergence
     * because of these if-statements, except if NY % _NTHREADS_ != 0.  The 6
     * if-statements for the boundary blocks are also necessary to ensure
     * independent block execution.
     *
     * NOTE: To minimize if-statements (and simplify code) the following
     *       requires that NX >= 5
     * */
    assert(NXp1 > 5);

    /* *
     * The general task order is (for each chunk slice along NZ):
     * 1.) Load ghosts from GMEM or tex3D into stencil (do this 7x, for each
     *     quantity)
     * 2.) Compute flux
     * 3.) Compute RHS for advection equations
     * */
    Stencil r, u, v, w, e, G, P;

    // LEFT GHOSTS
    if (iy < NY && 0 == ix)
    {
        for (uint_t iz = 3; iz < NZ+3; ++iz) // first and last 3 slices are zghosts
        {
            // 1.)
            _load_stencil_tex3D_X(r, texR, ix, iy, iz);
            _load_stencil_tex3D_X(u, texU, ix, iy, iz);
            _load_stencil_tex3D_X(v, texV, ix, iy, iz);
            _load_stencil_tex3D_X(w, texW, ix, iy, iz);
            _load_stencil_tex3D_X(e, texE, ix, iy, iz);
            _load_stencil_tex3D_X(G, texG, ix, iy, iz);
            _load_stencil_tex3D_X(P, texP, ix, iy, iz);
            _load_3ghosts_X(r.im3, r.im2, r.im1, ghostL.r, iy, iz-3+global_iz);
            _load_3ghosts_X(u.im3, u.im2, u.im1, ghostL.u, iy, iz-3+global_iz);
            _load_3ghosts_X(v.im3, v.im2, v.im1, ghostL.v, iy, iz-3+global_iz);
            _load_3ghosts_X(w.im3, w.im2, w.im1, ghostL.w, iy, iz-3+global_iz);
            _load_3ghosts_X(e.im3, e.im2, e.im1, ghostL.e, iy, iz-3+global_iz);
            _load_3ghosts_X(G.im3, G.im2, G.im1, ghostL.G, iy, iz-3+global_iz);
            _load_3ghosts_X(P.im3, P.im2, P.im1, ghostL.P, iy, iz-3+global_iz);

            // 2.)
            const uint_t flux_id = ID3(iy, ix, iz, NY, NXp1);
            _compute_flux_all(flux, r, u, v, w, e, G, P, flux_id, xtra_vel, xtra_Gm, xtra_Gp, xtra_Pm, xtra_Pp);
        }
    }
    else if (iy < NY && 1 == ix)
    {
        for (uint_t iz = 3; iz < NZ+3; ++iz)
        {
            // 1.)
            _load_stencil_tex3D_X(r, texR, ix, iy, iz);
            _load_stencil_tex3D_X(u, texU, ix, iy, iz);
            _load_stencil_tex3D_X(v, texV, ix, iy, iz);
            _load_stencil_tex3D_X(w, texW, ix, iy, iz);
            _load_stencil_tex3D_X(e, texE, ix, iy, iz);
            _load_stencil_tex3D_X(G, texG, ix, iy, iz);
            _load_stencil_tex3D_X(P, texP, ix, iy, iz);
            _load_2ghosts_X(r.im3, r.im2, 1, 2, ghostL.r, iy, iz-3+global_iz);
            _load_2ghosts_X(u.im3, u.im2, 1, 2, ghostL.u, iy, iz-3+global_iz);
            _load_2ghosts_X(v.im3, v.im2, 1, 2, ghostL.v, iy, iz-3+global_iz);
            _load_2ghosts_X(w.im3, w.im2, 1, 2, ghostL.w, iy, iz-3+global_iz);
            _load_2ghosts_X(e.im3, e.im2, 1, 2, ghostL.e, iy, iz-3+global_iz);
            _load_2ghosts_X(G.im3, G.im2, 1, 2, ghostL.G, iy, iz-3+global_iz);
            _load_2ghosts_X(P.im3, P.im2, 1, 2, ghostL.P, iy, iz-3+global_iz);


            /* _print_stencil(r, ix, iy, iz-3); */
        }
    }
    else if (iy < NY && 2 == ix)
    {
        for (uint_t iz = 3; iz < NZ+3; ++iz)
        {
            // 1.)
            _load_stencil_tex3D_X(r, texR, ix, iy, iz);
            _load_stencil_tex3D_X(u, texU, ix, iy, iz);
            _load_stencil_tex3D_X(v, texV, ix, iy, iz);
            _load_stencil_tex3D_X(w, texW, ix, iy, iz);
            _load_stencil_tex3D_X(e, texE, ix, iy, iz);
            _load_stencil_tex3D_X(G, texG, ix, iy, iz);
            _load_stencil_tex3D_X(P, texP, ix, iy, iz);
            _load_1ghost_X(r.im3, 2, ghostL.r, iy, iz-3+global_iz);
            _load_1ghost_X(u.im3, 2, ghostL.u, iy, iz-3+global_iz);
            _load_1ghost_X(v.im3, 2, ghostL.v, iy, iz-3+global_iz);
            _load_1ghost_X(w.im3, 2, ghostL.w, iy, iz-3+global_iz);
            _load_1ghost_X(e.im3, 2, ghostL.e, iy, iz-3+global_iz);
            _load_1ghost_X(G.im3, 2, ghostL.G, iy, iz-3+global_iz);
            _load_1ghost_X(P.im3, 2, ghostL.P, iy, iz-3+global_iz);


            /* _print_stencil(r, ix, iy, iz-3); */
        }
    }
    // RIGHT GHOSTS
    else if (iy < NY && NXp1-3 == ix)
    {
        for (uint_t iz = 3; iz < NZ+3; ++iz)
        {
            // 1.)
            _load_stencil_tex3D_X(r, texR, ix, iy, iz);
            _load_stencil_tex3D_X(u, texU, ix, iy, iz);
            _load_stencil_tex3D_X(v, texV, ix, iy, iz);
            _load_stencil_tex3D_X(w, texW, ix, iy, iz);
            _load_stencil_tex3D_X(e, texE, ix, iy, iz);
            _load_stencil_tex3D_X(G, texG, ix, iy, iz);
            _load_stencil_tex3D_X(P, texP, ix, iy, iz);
            _load_1ghost_X(r.ip2, 0, ghostR.r, iy, iz-3+global_iz);
            _load_1ghost_X(u.ip2, 0, ghostR.u, iy, iz-3+global_iz);
            _load_1ghost_X(v.ip2, 0, ghostR.v, iy, iz-3+global_iz);
            _load_1ghost_X(w.ip2, 0, ghostR.w, iy, iz-3+global_iz);
            _load_1ghost_X(e.ip2, 0, ghostR.e, iy, iz-3+global_iz);
            _load_1ghost_X(G.ip2, 0, ghostR.G, iy, iz-3+global_iz);
            _load_1ghost_X(P.ip2, 0, ghostR.P, iy, iz-3+global_iz);


            /* _print_stencil(r, ix, iy, iz-3); */
        }
    }
    else if (iy < NY && NXp1-2 == ix)
    {
        for (uint_t iz = 3; iz < NZ+3; ++iz)
        {
            // 1.)
            _load_stencil_tex3D_X(r, texR, ix, iy, iz);
            _load_stencil_tex3D_X(u, texU, ix, iy, iz);
            _load_stencil_tex3D_X(v, texV, ix, iy, iz);
            _load_stencil_tex3D_X(w, texW, ix, iy, iz);
            _load_stencil_tex3D_X(e, texE, ix, iy, iz);
            _load_stencil_tex3D_X(G, texG, ix, iy, iz);
            _load_stencil_tex3D_X(P, texP, ix, iy, iz);
            _load_2ghosts_X(r.ip1, r.ip2, 0, 1, ghostR.r, iy, iz-3+global_iz);
            _load_2ghosts_X(u.ip1, u.ip2, 0, 1, ghostR.u, iy, iz-3+global_iz);
            _load_2ghosts_X(v.ip1, v.ip2, 0, 1, ghostR.v, iy, iz-3+global_iz);
            _load_2ghosts_X(w.ip1, w.ip2, 0, 1, ghostR.w, iy, iz-3+global_iz);
            _load_2ghosts_X(e.ip1, e.ip2, 0, 1, ghostR.e, iy, iz-3+global_iz);
            _load_2ghosts_X(G.ip1, G.ip2, 0, 1, ghostR.G, iy, iz-3+global_iz);
            _load_2ghosts_X(P.ip1, P.ip2, 0, 1, ghostR.P, iy, iz-3+global_iz);


            /* _print_stencil(r, ix, iy, iz-3); */
        }
    }
    else if (iy < NY && NXp1-1 == ix)
    {
        for (uint_t iz = 3; iz < NZ+3; ++iz)
        {
            // 1.)
            _load_stencil_tex3D_X(r, texR, ix, iy, iz);
            _load_stencil_tex3D_X(u, texU, ix, iy, iz);
            _load_stencil_tex3D_X(v, texV, ix, iy, iz);
            _load_stencil_tex3D_X(w, texW, ix, iy, iz);
            _load_stencil_tex3D_X(e, texE, ix, iy, iz);
            _load_stencil_tex3D_X(G, texG, ix, iy, iz);
            _load_stencil_tex3D_X(P, texP, ix, iy, iz);
            _load_3ghosts_X(r.i, r.ip1, r.ip2, ghostR.r, iy, iz-3+global_iz);
            _load_3ghosts_X(u.i, u.ip1, u.ip2, ghostR.u, iy, iz-3+global_iz);
            _load_3ghosts_X(v.i, v.ip1, v.ip2, ghostR.v, iy, iz-3+global_iz);
            _load_3ghosts_X(w.i, w.ip1, w.ip2, ghostR.w, iy, iz-3+global_iz);
            _load_3ghosts_X(e.i, e.ip1, e.ip2, ghostR.e, iy, iz-3+global_iz);
            _load_3ghosts_X(G.i, G.ip1, G.ip2, ghostR.G, iy, iz-3+global_iz);
            _load_3ghosts_X(P.i, P.ip1, P.ip2, ghostR.P, iy, iz-3+global_iz);


            /* _print_stencil(r, ix, iy, iz-3); */
        }
    }
    // INTERIOR (MUST be last if!)
    else if (iy < NY && (3 <= ix && ix < NXp1-3))
    {
        for (uint_t iz = 3; iz < NZ+3; ++iz)
        {
            // 1.)
            _load_stencil_tex3D_X(r, texR, ix, iy, iz);
            _load_stencil_tex3D_X(u, texU, ix, iy, iz);
            _load_stencil_tex3D_X(v, texV, ix, iy, iz);
            _load_stencil_tex3D_X(w, texW, ix, iy, iz);
            _load_stencil_tex3D_X(e, texE, ix, iy, iz);
            _load_stencil_tex3D_X(G, texG, ix, iy, iz);
            _load_stencil_tex3D_X(P, texP, ix, iy, iz);


            /* _print_stencil(r, ix, iy, iz-3); */
        }
    }



    /* if (ix < NXp1 && iy < NY) */
    /* { */
    /*     // Process NZ slices of current chunk */
    /*     // iz = 0, 1, 2: left zghost slices */
    /*     // iz = NZ+3, NZ+4, NZ+5: right zghost slices */
    /*     for (uint_t iz = 3; iz < NZ+3; ++iz) */
    /*     { */
    /*         /1* * */
    /*          * 1.) Get cell values */
    /*          * 2.) Reconstruct face values (in primitive variables) */
    /*          * 3.) Compute characteristic velocities */
    /*          * 4.) Compute 7 flux contributions */
    /*          * 5.) Compute right hand side for the advection equations */
    /*          * *1/ */

    /*         /////////////////////////////////////////////////////////////////// */
    /*         // 1.) Load data */
    /*         /////////////////////////////////////////////////////////////////// */
    /*         Real rm3, rm2, rm1, rp1, rp2, rp3; */
    /*         _xfetch_data(texR, ghostL.r, ghostR.r, ix, iy, iz, global_iz, NXp1, NY, rm3, rm2, rm1, rp1, rp2, rp3); */
    /*         assert(rm3 > 0); assert(rm2 > 0); assert(rm1 > 0); assert(rp1 > 0); assert(rp2 > 0); assert(rp3 > 0); */

    /*         Real um3, um2, um1, up1, up2, up3; */
    /*         _xfetch_data(texU, ghostL.u, ghostR.u, ix, iy, iz, global_iz, NXp1, NY, um3, um2, um1, up1, up2, up3); */

    /*         Real vm3, vm2, vm1, vp1, vp2, vp3; */
    /*         _xfetch_data(texV, ghostL.v, ghostR.v, ix, iy, iz, global_iz, NXp1, NY, vm3, vm2, vm1, vp1, vp2, vp3); */

    /*         Real wm3, wm2, wm1, wp1, wp2, wp3; */
    /*         _xfetch_data(texW, ghostL.w, ghostR.w, ix, iy, iz, global_iz, NXp1, NY, wm3, wm2, wm1, wp1, wp2, wp3); */

    /*         Real em3, em2, em1, ep1, ep2, ep3; */
    /*         _xfetch_data(texE, ghostL.e, ghostR.e, ix, iy, iz, global_iz, NXp1, NY, em3, em2, em1, ep1, ep2, ep3); */
    /*         assert(em3 > 0); assert(em2 > 0); assert(em1 > 0); assert(ep1 > 0); assert(ep2 > 0); assert(ep3 > 0); */

    /*         Real Gm3, Gm2, Gm1, Gp1, Gp2, Gp3; */
    /*         _xfetch_data(texG, ghostL.G, ghostR.G, ix, iy, iz, global_iz, NXp1, NY, Gm3, Gm2, Gm1, Gp1, Gp2, Gp3); */
    /*         assert(Gm3 > 0); assert(Gm2 > 0); assert(Gm1 > 0); assert(Gp1 > 0); assert(Gp2 > 0); assert(Gp3 > 0); */

    /*         Real Pm3, Pm2, Pm1, Pp1, Pp2, Pp3; */
    /*         _xfetch_data(texP, ghostL.P, ghostR.P, ix, iy, iz, global_iz, NXp1, NY, Pm3, Pm2, Pm1, Pp1, Pp2, Pp3); */
    /*         assert(Pm3 >= 0); assert(Pm2 >= 0); assert(Pm1 >= 0); assert(Pp1 >= 0); assert(Pp2 >= 0); assert(Pp3 >= 0); */

    /*         /////////////////////////////////////////////////////////////////// */
    /*         // 2.) Reconstruction of primitive values, using WENO5/3 */
    /*         /////////////////////////////////////////////////////////////////// */
    /*         // Reconstruct primitive value p at face f, using WENO5/3 */
    /*         // rho */
    /*         const Real rp = _weno_pluss_clipped(rm2, rm1, rp1, rp2, rp3); */
    /*         const Real rm = _weno_minus_clipped(rm3, rm2, rm1, rp1, rp2); */
    /*         assert(!isnan(rp)); assert(!isnan(rm)); */
    /*         // u (convert primitive variable u = (rho*u) / rho) */
    /*         um3 /= rm3; um2 /= rm2; um1 /= rm1; up1 /= rp1; up2 /= rp2; up3 /= rp3; */
    /*         const Real up = _weno_pluss_clipped(um2, um1, up1, up2, up3); */
    /*         const Real um = _weno_minus_clipped(um3, um2, um1, up1, up2); */
    /*         assert(!isnan(up)); assert(!isnan(um)); */
    /*         // v (convert primitive variable v = (rho*v) / rho) */
    /*         vm3 /= rm3; vm2 /= rm2; vm1 /= rm1; vp1 /= rp1; vp2 /= rp2; vp3 /= rp3; */
    /*         const Real vp = _weno_pluss_clipped(vm2, vm1, vp1, vp2, vp3); */
    /*         const Real vm = _weno_minus_clipped(vm3, vm2, vm1, vp1, vp2); */
    /*         assert(!isnan(vp)); assert(!isnan(vm)); */
    /*         // w (convert primitive variable w = (rho*w) / rho) */
    /*         wm3 /= rm3; wm2 /= rm2; wm1 /= rm1; wp1 /= rp1; wp2 /= rp2; wp3 /= rp3; */
    /*         const Real wp = _weno_pluss_clipped(wm2, wm1, wp1, wp2, wp3); */
    /*         const Real wm = _weno_minus_clipped(wm3, wm2, wm1, wp1, wp2); */
    /*         assert(!isnan(wp)); assert(!isnan(wm)); */
    /*         // p (convert primitive variable p = (e - 0.5*rho*(u*u + v*v + w*w) - P) / G */
    /*         const Real pm3 = (em3 - 0.5f*rm3*(um3*um3 + vm3*vm3 + wm3*wm3) - Pm3) / Gm3; */
    /*         const Real pm2 = (em2 - 0.5f*rm2*(um2*um2 + vm2*vm2 + wm2*wm2) - Pm2) / Gm2; */
    /*         const Real pm1 = (em1 - 0.5f*rm1*(um1*um1 + vm1*vm1 + wm1*wm1) - Pm1) / Gm1; */
    /*         const Real pp1 = (ep1 - 0.5f*rp1*(up1*up1 + vp1*vp1 + wp1*wp1) - Pp1) / Gp1; */
    /*         const Real pp2 = (ep2 - 0.5f*rp2*(up2*up2 + vp2*vp2 + wp2*wp2) - Pp2) / Gp2; */
    /*         const Real pp3 = (ep3 - 0.5f*rp3*(up3*up3 + vp3*vp3 + wp3*wp3) - Pp3) / Gp3; */
    /*         const Real pp = _weno_pluss_clipped(pm2, pm1, pp1, pp2, pp3); */
    /*         const Real pm = _weno_minus_clipped(pm3, pm2, pm1, pp1, pp2); */
    /*         assert(!isnan(pp)); assert(!isnan(pm)); */
    /*         // G */
    /*         const Real Gp = _weno_pluss_clipped(Gm2, Gm1, Gp1, Gp2, Gp3); */
    /*         const Real Gm = _weno_minus_clipped(Gm3, Gm2, Gm1, Gp1, Gp2); */
    /*         assert(!isnan(Gp)); assert(!isnan(Gm)); */
    /*         // P */
    /*         const Real Pp = _weno_pluss_clipped(Pm2, Pm1, Pp1, Pp2, Pp3); */
    /*         const Real Pm = _weno_minus_clipped(Pm3, Pm2, Pm1, Pp1, Pp2); */
    /*         assert(!isnan(Pp)); assert(!isnan(Pm)); */

    /*         /////////////////////////////////////////////////////////////////// */
    /*         // 3.) Einfeldt characteristic velocities */
    /*         /////////////////////////////////////////////////////////////////// */
    /*         Real sm, sp; */
    /*         _char_vel_einfeldt(rm, rp, um, up, pm, pp, Gm, Gp, Pm, Pp, sm, sp); */
    /*         const Real ss = _char_vel_star(rm, rp, um, up, pm, pp, sm, sp); */
    /*         assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss)); */

    /*         /////////////////////////////////////////////////////////////////// */
    /*         // 4.) Compute HLLC fluxes */
    /*         /////////////////////////////////////////////////////////////////// */
    /*         const Real fr = _hllc_rho(rm, rp, um, up, sm, sp, ss); */
    /*         const Real fu = _hllc_pvel(rm, rp, um, up, pm, pp, sm, sp, ss); */
    /*         const Real fv = _hllc_vel(rm, rp, vm, vp, um, up, sm, sp, ss); */
    /*         const Real fw = _hllc_vel(rm, rp, wm, wp, um, up, sm, sp, ss); */
    /*         const Real fe = _hllc_e(rm, rp, um, up, vm, vp, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss); */
    /*         const Real fG = _hllc_rho(Gm, Gp, um, up, sm, sp, ss); */
    /*         const Real fP = _hllc_rho(Pm, Pp, um, up, sm, sp, ss); */
    /*         assert(!isnan(fr)); assert(!isnan(fu)); assert(!isnan(fv)); assert(!isnan(fw)); assert(!isnan(fe)); assert(!isnan(fG)); assert(!isnan(fP)); */

    /*         const uint_t idx = ID3(iy, ix, iz, NY, NXp1); */
    /*         flux.r[idx] = fr; */
    /*         flux.u[idx] = fu; */
    /*         flux.v[idx] = fv; */
    /*         flux.w[idx] = fw; */
    /*         flux.e[idx] = fe; */
    /*         flux.G[idx] = fG; */
    /*         flux.P[idx] = fP; */

    /*         /////////////////////////////////////////////////////////////////// */
    /*         // 5.) RHS for advection equations */
    /*         /////////////////////////////////////////////////////////////////// */
    /*         xtra_vel[idx] = _extraterm_hllc_vel(um, up, Gm, Gp, Pm, Pp, sm, sp, ss); */
    /*         xtra_Gm[idx]  = Gm; */
    /*         xtra_Gp[idx]  = Gp; */
    /*         xtra_Pm[idx]  = Pm; */
    /*         xtra_Pp[idx]  = Pp; */
    /*     } */
    /* } */
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
            // 1.) Load data
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
            // 1.) Load data
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
        const Real factor6 = 1.0f / 6.0f;
        for (uint_t iz = 0; iz < NZ; ++iz)
        {
            const uint_t idx = ID3(ix, iy, iz, NX, NY);

            _fetch_flux(ix, iy, iz, NX, NY, xflux.r, yflux.r, zflux.r, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_r = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.r[idx] = a*tmp.r[idx] - rhs_r;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.u, yflux.u, zflux.u, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_u = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.u[idx] = a*tmp.u[idx] - rhs_u;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.v, yflux.v, zflux.v, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_v = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.v[idx] = a*tmp.v[idx] - rhs_v;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.w, yflux.w, zflux.w, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_w = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.w[idx] = a*tmp.w[idx] - rhs_w;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.e, yflux.e, zflux.e, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_e = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.e[idx] = a*tmp.e[idx] - rhs_e;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.G, yflux.G, zflux.G, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_G = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm   - divU[idx] * sumG[idx] * factor6);
            rhs.G[idx] = a*tmp.G[idx] - rhs_G;

            _fetch_flux(ix, iy, iz, NX, NY, xflux.P, yflux.P, zflux.P, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_P = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm   - divU[idx] * sumP[idx] * factor6);
            rhs.P[idx] = a*tmp.P[idx] - rhs_P;
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
            assert(tmp.r[idx] > 0);
            assert(tmp.e[idx] > 0);
            assert(tmp.G[idx] > 0);
            assert(tmp.P[idx] >= 0);
            /* if (tmp.P[idx] < 0) */
            /*     printf("(%d, %d, %d):\trhs.P = %f, tmp.P = %f, P = %f\n", ix, iy, iz, rhs.P[idx], tmp.P[idx], P); */
        }
    }
}


__global__
void _maxSOS(const uint_t NX, const uint_t NY, const uint_t NZ, int* g_maxSOS)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    const uint_t loc_idx = blockDim.x * threadIdx.y + threadIdx.x;
    __shared__ Real block_sos[_NTHREADS_];
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
            for (int i = 1; i < _NTHREADS_; ++i)
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
#ifndef _MUTE_GPU_
        devPtrSet xghostL(d_xgl);
        devPtrSet xghostR(d_xgr);
        devPtrSet xflux(d_xflux);

        {
            const uint_t XSize = BSX_GPU + 1;
            const dim3 grid(XSize, (BSY_GPU + _NTHREADS_ -1) / _NTHREADS_, 1);
            const dim3 blocks(1, _NTHREADS_, 1);

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
            const dim3 grid(XSize, (BSY_GPU + _NTHREADS_ -1) / _NTHREADS_, 1);
            const dim3 blocks(1, _NTHREADS_, 1);
            GPUtimer xextra;
            xextra.start(stream1);
            _xextraterm_hllc<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            xextra.stop(stream1);
            xextra.print("[_xextraterm Kernel]: ");
        }
#endif
    }


    void GPU::yflux(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH, const uint_t global_iz)
    {
#ifndef _MUTE_GPU_
        devPtrSet yghostL(d_ygl);
        devPtrSet yghostR(d_ygr);
        devPtrSet yflux(d_yflux);

        {
            const uint_t YSize = BSY_GPU + 1;
            const dim3 grid((BSX_GPU + _NTHREADS_ -1) / _NTHREADS_, YSize, 1);
            const dim3 blocks(_NTHREADS_, 1, 1);

            GPUtimer kernel;
            kernel.start(stream1);
            _yflux<<<grid, blocks, 0, stream1>>>(BSX_GPU, YSize, CHUNK_WIDTH, global_iz, yghostL, yghostR, yflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            kernel.stop(stream1);
            kernel.print("[_yflux Kernel]: ");
        }

        {
            const uint_t YSize = BSY_GPU;
            const dim3 grid((BSX_GPU + _NTHREADS_ -1) / _NTHREADS_, YSize, 1);
            const dim3 blocks(_NTHREADS_, 1, 1);

            GPUtimer yextra;
            yextra.start(stream1);
            _yextraterm_hllc<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            yextra.stop(stream1);
            yextra.print("[_yextraterm Kernel]: ");
        }
#endif
    }


    void GPU::zflux(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH)
    {
#ifndef _MUTE_GPU_
        devPtrSet zflux(d_zflux);

        {
            const uint_t ZSize = CHUNK_WIDTH + 1;
            const dim3 grid((BSX_GPU + _NTHREADS_ -1) / _NTHREADS_, BSY_GPU, 1);
            const dim3 blocks(_NTHREADS_, 1, 1);

            GPUtimer kernel;
            kernel.start(stream1);
            _zflux<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, ZSize, zflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            kernel.stop(stream1);
            kernel.print("[_zflux Kernel]: ");
        }

        {
            const dim3 grid((BSX_GPU + _NTHREADS_ -1) / _NTHREADS_, BSY_GPU, 1);
            const dim3 blocks(_NTHREADS_, 1, 1);

            GPUtimer zextra;
            zextra.start(stream1);
            _zextraterm_hllc<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
            zextra.stop(stream1);
            zextra.print("[_zextraterm Kernel]: ");
        }
#endif
    }


    void GPU::divergence(const Real a, const Real dtinvh, const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH)
    {
#ifndef _MUTE_GPU_
        cudaStreamWaitEvent(stream1, h2d_tmp_completed, 0);

        devPtrSet xflux(d_xflux);
        devPtrSet yflux(d_yflux);
        devPtrSet zflux(d_zflux);
        devPtrSet rhs(d_rhs);
        devPtrSet tmp(d_tmp);

        const dim3 grid((BSX_GPU + _NTHREADS_ -1) / _NTHREADS_, BSY_GPU, 1);
        const dim3 blocks(_NTHREADS_, 1, 1);

        GPUtimer kernel;
        kernel.start(stream1);
        _divergence<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, xflux, yflux, zflux, rhs, a, dtinvh, tmp, d_sumG, d_sumP, d_divU);
        kernel.stop(stream1);
        kernel.print("[_divergence Kernel]: ");

        cudaEventRecord(divergence_completed, stream1);
#endif
    }


    void GPU::update(const Real b, const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH)
    {
#ifndef _MUTE_GPU_
        devPtrSet tmp(d_tmp);
        devPtrSet rhs(d_rhs);

        const dim3 grid((BSX_GPU + _NTHREADS_ -1) / _NTHREADS_, BSY_GPU, 1);
        const dim3 blocks(_NTHREADS_, 1, 1);

        GPUtimer kernel;
        kernel.start(stream1);
        _update<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, b, tmp, rhs);
        kernel.stop(stream1);
        kernel.print("[_update Kernel]: ");

        cudaEventRecord(update_completed, stream1);
#endif
    }


    void GPU::MaxSpeedOfSound(const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t CHUNK_WIDTH)
    {
#ifndef _MUTE_GPU_
        const dim3 grid((BSX_GPU + _NTHREADS_ -1) / _NTHREADS_, BSY_GPU, 1);
        const dim3 blocks(_NTHREADS_, 1, 1);

        GPUtimer kernel;
        kernel.start(stream1);
        _maxSOS<<<grid, blocks, 0, stream1>>>(BSX_GPU, BSY_GPU, CHUNK_WIDTH, d_maxSOS);
        kernel.stop(stream1);
        kernel.print("[_maxSOS Kernel]: ");
#endif
    }


    ///////////////////////////////////////////////////////////////////////////
    // TEST SECTION
    ///////////////////////////////////////////////////////////////////////////
    void GPU::TestKernel()
    {
        devPtrSet xghostL(d_xgl);
        devPtrSet xghostR(d_xgr);
        devPtrSet xflux(d_xflux);

        {
            const uint_t NXp1 = NodeBlock::sizeX + 1;
            const uint_t NY   = NodeBlock::sizeY;
            const uint_t NZ   = NodeBlock::sizeZ;

            const dim3 grid(NXp1, (NY + _NTHREADS_ - 1) / _NTHREADS_, 1);
            const dim3 blocks(1, _NTHREADS_, 1);

            GPUtimer kernel;
            kernel.start();
            _xflux<<<grid, blocks>>>(NXp1, NY, NZ, 0, xghostL, xghostR, xflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
            kernel.stop();
            kernel.print("[Testing Kernel]: ");
        }


    }
}


///////////////////////////////////////////////////////////////////////////////
//                                   UTILS                                   //
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


extern "C"
{
    void GPU::bind_textures()
    {
#ifndef _MUTE_GPU_
        _bindTexture(&texR, d_SOAin[0]);
        _bindTexture(&texU, d_SOAin[1]);
        _bindTexture(&texV, d_SOAin[2]);
        _bindTexture(&texW, d_SOAin[3]);
        _bindTexture(&texE, d_SOAin[4]);
        _bindTexture(&texG, d_SOAin[5]);
        _bindTexture(&texP, d_SOAin[6]);
#endif
    }


    void GPU::unbind_textures()
    {
#ifndef _MUTE_GPU_
        cudaUnbindTexture(&texR);
        cudaUnbindTexture(&texU);
        cudaUnbindTexture(&texV);
        cudaUnbindTexture(&texW);
        cudaUnbindTexture(&texE);
        cudaUnbindTexture(&texG);
        cudaUnbindTexture(&texP);
#endif
    }
}

/* File        : GPUkernels_dev.cu */
/* Maintainer  : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Tue 29 Jul 2014 11:25:42 AM CEST */
/* Modified    : Tue 05 Aug 2014 02:22:30 PM CEST */
/* Description : Development stuff, which is taken out of the main kernel
 *               source GPUkernels.cu.  This source is not used for any
 *               compilation or use, it merely is a trunk to keep some
 *               experimental kernel source, the main kernel source code
 *               remains clean. */


#define ARBITRARY_SLICE_DIM
#ifdef ARBITRARY_SLICE_DIM
// BUGGY !!!!!!!!!!!!!!! + likely to yield less performance as if slice
// dimensions would be ineger multiples of TILE_DIM.
__global__
void _xextraterm_hllc(const uint_t nslices,
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
    uint_t ix = blockIdx.x * TILE_DIM + threadIdx.x;
    uint_t iy = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ Real smem[TILE_DIM*(TILE_DIM+1)];

    if (ix < NX && iy < NY)
    {
        // compute this blocks actual stride for shared memory access.  This is
        // required for arbitray dimensions. Stride for fastest moving index
        // must be stride+1 to avoid bank conflicts.
        const uint_t NXsmem = (NX < TILE_DIM*(blockIdx.x+1)) ? (NX - blockIdx.x*TILE_DIM + 1) : (TILE_DIM+1);
        const uint_t NYsmem = (NY < TILE_DIM*(blockIdx.y+1)) ? (NY - blockIdx.y*TILE_DIM) : TILE_DIM;

        // transpose
        const uint_t iyT = blockIdx.y * TILE_DIM + threadIdx.x;
        const uint_t ixT = blockIdx.x * TILE_DIM + threadIdx.y;

        for (uint_t iz = 0; iz < nslices; ++iz)
        {
            // G
            smem[threadIdx.x*NYsmem + threadIdx.y] = Gp[ID3(iyT,ixT,iz,NY,NXP1)] + Gm[ID3(iyT,(ixT+1),iz,NY,NXP1)];
            __syncthreads();
            sumG[ID3(ix,iy,iz,NX,NY)] = smem[threadIdx.y*NXsmem + threadIdx.x];
            __syncthreads();

            // P
            smem[threadIdx.x*NYsmem + threadIdx.y] = Pp[ID3(iyT,ixT,iz,NY,NXP1)] + Pm[ID3(iyT,(ixT+1),iz,NY,NXP1)];
            __syncthreads();
            sumP[ID3(ix,iy,iz,NX,NY)] = smem[threadIdx.y*NXsmem + threadIdx.x];
            __syncthreads();

            // Velocity on cell faces
            smem[threadIdx.x*NYsmem + threadIdx.y] = vel[ID3(iyT,(ixT+1),iz,NY,NXP1)] - vel[ID3(iyT,ixT,iz,NY,NXP1)];
            __syncthreads();
            divU[ID3(ix,iy,iz,NX,NY)] = smem[threadIdx.y*NXsmem + threadIdx.x];
            __syncthreads();
        }
    }
}
#endif


__global__
void _xflux(const uint_t nslices, const uint_t global_iz,
        devPtrSet ghostL, devPtrSet ghostR, devPtrSet flux,
        Real * const xtra_vel,
        Real * const xtra_Gm, Real * const xtra_Gp,
        Real * const xtra_Pm, Real * const xtra_Pp)
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
     * */
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    /* *
     * The following code requires 1D thread blocks with dimension
     * dim3(_NTHREADS_, 1, 1)
     * on a 2D grid with dimension
     * dim3((NXP1 + _NTHREADS_ - 1) / _NTHREADS_, NY, 1)
     *
     * The load of ghosts is organized into a switch block with 6 cases. Blocks
     * affected by this are the first three on the left boundary and the last
     * three on the right. Given the layout of thread blocks above, warps do
     * not diverge because of the switch.
     *
     * NOTE: To minimize the switch cases to 6 (and simplify code) the
     * following requires that NX >= 5
     * */
    assert(NXP1 > 5);

#if 1
    if (ix < NXP1 && iy < NY)
    {
        Stencil r, u, v, w, e, G, P;
        Stencil p; // for reconstruction
        for (uint_t iz = 3; iz < nslices+3; ++iz) // first and last 3 slices are zghosts
        {
            /* *
             * The general task order is (for each chunk slice along NZ):
             * 1.) Load ghosts from GMEM or tex3D into stencil (do this 7x, for each
             *     quantity)
             * 2.) Reconstruct primitive values using WENO5/WENO3
             * 3.) Compute characteristic velocities
             * 4.) Compute fluxes
             * 5.) Compute RHS for advection of G and P
             * */

            // 1.)
#if 1
            // more conditionals
            _read_stencil_X(r, texR, ghostL.r, ghostR.r, ix, iy, iz, global_iz);
            _read_stencil_X(u, texU, ghostL.u, ghostR.u, ix, iy, iz, global_iz);
            _read_stencil_X(v, texV, ghostL.v, ghostR.v, ix, iy, iz, global_iz);
            _read_stencil_X(w, texW, ghostL.w, ghostR.w, ix, iy, iz, global_iz);
            _read_stencil_X(e, texE, ghostL.e, ghostR.e, ix, iy, iz, global_iz);
            _read_stencil_X(G, texG, ghostL.G, ghostR.G, ix, iy, iz, global_iz);
            _read_stencil_X(P, texP, ghostL.P, ghostR.P, ix, iy, iz, global_iz);
#else
            // less conditionals, additional tex3D fetches
            _load_stencil_tex3D_X(r, texR, ix, iy, iz);
            _load_stencil_tex3D_X(u, texU, ix, iy, iz);
            _load_stencil_tex3D_X(v, texV, ix, iy, iz);
            _load_stencil_tex3D_X(w, texW, ix, iy, iz);
            _load_stencil_tex3D_X(e, texE, ix, iy, iz);
            _load_stencil_tex3D_X(G, texG, ix, iy, iz);
            _load_stencil_tex3D_X(P, texP, ix, iy, iz);
            switch (ix)
            {
                case 0:
                    _load_3ghosts_X(r.im3, r.im2, r.im1, ghostL.r, iy, iz-3+global_iz);
                    _load_3ghosts_X(u.im3, u.im2, u.im1, ghostL.u, iy, iz-3+global_iz);
                    _load_3ghosts_X(v.im3, v.im2, v.im1, ghostL.v, iy, iz-3+global_iz);
                    _load_3ghosts_X(w.im3, w.im2, w.im1, ghostL.w, iy, iz-3+global_iz);
                    _load_3ghosts_X(e.im3, e.im2, e.im1, ghostL.e, iy, iz-3+global_iz);
                    _load_3ghosts_X(G.im3, G.im2, G.im1, ghostL.G, iy, iz-3+global_iz);
                    _load_3ghosts_X(P.im3, P.im2, P.im1, ghostL.P, iy, iz-3+global_iz);
                    break;
                case 1:
                    _load_2ghosts_X(r.im3, r.im2, 1, 2, ghostL.r, iy, iz-3+global_iz);
                    _load_2ghosts_X(u.im3, u.im2, 1, 2, ghostL.u, iy, iz-3+global_iz);
                    _load_2ghosts_X(v.im3, v.im2, 1, 2, ghostL.v, iy, iz-3+global_iz);
                    _load_2ghosts_X(w.im3, w.im2, 1, 2, ghostL.w, iy, iz-3+global_iz);
                    _load_2ghosts_X(e.im3, e.im2, 1, 2, ghostL.e, iy, iz-3+global_iz);
                    _load_2ghosts_X(G.im3, G.im2, 1, 2, ghostL.G, iy, iz-3+global_iz);
                    _load_2ghosts_X(P.im3, P.im2, 1, 2, ghostL.P, iy, iz-3+global_iz);
                    break;
                case 2:
                    _load_1ghost_X(r.im3, 2, ghostL.r, iy, iz-3+global_iz);
                    _load_1ghost_X(u.im3, 2, ghostL.u, iy, iz-3+global_iz);
                    _load_1ghost_X(v.im3, 2, ghostL.v, iy, iz-3+global_iz);
                    _load_1ghost_X(w.im3, 2, ghostL.w, iy, iz-3+global_iz);
                    _load_1ghost_X(e.im3, 2, ghostL.e, iy, iz-3+global_iz);
                    _load_1ghost_X(G.im3, 2, ghostL.G, iy, iz-3+global_iz);
                    _load_1ghost_X(P.im3, 2, ghostL.P, iy, iz-3+global_iz);
                    break;
                case (NXP1-3):
                    _load_1ghost_X(r.ip2, 0, ghostR.r, iy, iz-3+global_iz);
                    _load_1ghost_X(u.ip2, 0, ghostR.u, iy, iz-3+global_iz);
                    _load_1ghost_X(v.ip2, 0, ghostR.v, iy, iz-3+global_iz);
                    _load_1ghost_X(w.ip2, 0, ghostR.w, iy, iz-3+global_iz);
                    _load_1ghost_X(e.ip2, 0, ghostR.e, iy, iz-3+global_iz);
                    _load_1ghost_X(G.ip2, 0, ghostR.G, iy, iz-3+global_iz);
                    _load_1ghost_X(P.ip2, 0, ghostR.P, iy, iz-3+global_iz);
                    break;
                case (NXP1-2):
                    _load_2ghosts_X(r.ip1, r.ip2, 0, 1, ghostR.r, iy, iz-3+global_iz);
                    _load_2ghosts_X(u.ip1, u.ip2, 0, 1, ghostR.u, iy, iz-3+global_iz);
                    _load_2ghosts_X(v.ip1, v.ip2, 0, 1, ghostR.v, iy, iz-3+global_iz);
                    _load_2ghosts_X(w.ip1, w.ip2, 0, 1, ghostR.w, iy, iz-3+global_iz);
                    _load_2ghosts_X(e.ip1, e.ip2, 0, 1, ghostR.e, iy, iz-3+global_iz);
                    _load_2ghosts_X(G.ip1, G.ip2, 0, 1, ghostR.G, iy, iz-3+global_iz);
                    _load_2ghosts_X(P.ip1, P.ip2, 0, 1, ghostR.P, iy, iz-3+global_iz);
                    break;
                case (NXP1-1):
                    _load_3ghosts_X(r.i, r.ip1, r.ip2, ghostR.r, iy, iz-3+global_iz);
                    _load_3ghosts_X(u.i, u.ip1, u.ip2, ghostR.u, iy, iz-3+global_iz);
                    _load_3ghosts_X(v.i, v.ip1, v.ip2, ghostR.v, iy, iz-3+global_iz);
                    _load_3ghosts_X(w.i, w.ip1, w.ip2, ghostR.w, iy, iz-3+global_iz);
                    _load_3ghosts_X(e.i, e.ip1, e.ip2, ghostR.e, iy, iz-3+global_iz);
                    _load_3ghosts_X(G.i, G.ip1, G.ip2, ghostR.G, iy, iz-3+global_iz);
                    _load_3ghosts_X(P.i, P.ip1, P.ip2, ghostR.P, iy, iz-3+global_iz);
                    break;
            } // end switch
#endif
            assert(r > 0);
            assert(e > 0);
            assert(G > 0);
            assert(P >= 0);

            // 2.)
            // rho
            const Real rp = _weno_pluss_clipped(r.im2, r.im1, r.i, r.ip1, r.ip2);
            const Real rm = _weno_minus_clipped(r.im3, r.im2, r.im1, r.i, r.ip1);
            assert(!isnan(rp)); assert(!isnan(rm));
            // u (convert primitive variable u = (rho*u) / rho)
            u.im3 /= r.im3;
            u.im2 /= r.im2;
            u.im1 /= r.im1;
            u.i   /= r.i;
            u.ip1 /= r.ip1;
            u.ip2 /= r.ip2;
            const Real up = _weno_pluss_clipped(u.im2, u.im1, u.i, u.ip1, u.ip2);
            const Real um = _weno_minus_clipped(u.im3, u.im2, u.im1, u.i, u.ip1);
            assert(!isnan(up)); assert(!isnan(um));
            // v (convert primitive variable v = (rho*v) / rho)
            v.im3 /= r.im3;
            v.im2 /= r.im2;
            v.im1 /= r.im1;
            v.i   /= r.i;
            v.ip1 /= r.ip1;
            v.ip2 /= r.ip2;
            const Real vp = _weno_pluss_clipped(v.im2, v.im1, v.i, v.ip1, v.ip2);
            const Real vm = _weno_minus_clipped(v.im3, v.im2, v.im1, v.i, v.ip1);
            assert(!isnan(vp)); assert(!isnan(vm));
            // w (convert primitive variable w = (rho*w) / rho)
            w.im3 /= r.im3;
            w.im2 /= r.im2;
            w.im1 /= r.im1;
            w.i   /= r.i;
            w.ip1 /= r.ip1;
            w.ip2 /= r.ip2;
            const Real wp = _weno_pluss_clipped(w.im2, w.im1, w.i, w.ip1, w.ip2);
            const Real wm = _weno_minus_clipped(w.im3, w.im2, w.im1, w.i, w.ip1);
            assert(!isnan(wp)); assert(!isnan(wm));
            // p (convert primitive variable p = (e - 0.5*rho*(u*u + v*v + w*w) - P) / G
            p.im3 = (e.im3 - 0.5f*r.im3*(u.im3*u.im3 + v.im3*v.im3 + w.im3*w.im3) - P.im3) / G.im3;
            p.im2 = (e.im2 - 0.5f*r.im2*(u.im2*u.im2 + v.im2*v.im2 + w.im2*w.im2) - P.im2) / G.im2;
            p.im1 = (e.im1 - 0.5f*r.im1*(u.im1*u.im1 + v.im1*v.im1 + w.im1*w.im1) - P.im1) / G.im1;
            p.i   = (e.i   - 0.5f*r.i*(u.i*u.i       + v.i*v.i     + w.i*w.i)     - P.i)   / G.i;
            p.ip1 = (e.ip1 - 0.5f*r.ip1*(u.ip1*u.ip1 + v.ip1*v.ip1 + w.ip1*w.ip1) - P.ip1) / G.ip1;
            p.ip2 = (e.ip2 - 0.5f*r.ip2*(u.ip2*u.ip2 + v.ip2*v.ip2 + w.ip2*w.ip2) - P.ip2) / G.ip2;
            const Real pp = _weno_pluss_clipped(p.im2, p.im1, p.i, p.ip1, p.ip2);
            const Real pm = _weno_minus_clipped(p.im3, p.im2, p.im1, p.i, p.ip1);
            assert(!isnan(pp)); assert(!isnan(pm));
            // G
            const Real Gp = _weno_pluss_clipped(G.im2, G.im1, G.i, G.ip1, G.ip2);
            const Real Gm = _weno_minus_clipped(G.im3, G.im2, G.im1, G.i, G.ip1);
            assert(!isnan(Gp)); assert(!isnan(Gm));
            // P
            const Real Pp = _weno_pluss_clipped(P.im2, P.im1, P.i, P.ip1, P.ip2);
            const Real Pm = _weno_minus_clipped(P.im3, P.im2, P.im1, P.i, P.ip1);
            assert(!isnan(Pp)); assert(!isnan(Pm));

            // 3.)
            Real sm, sp;
            _char_vel_einfeldt(rm, rp, um, up, pm, pp, Gm, Gp, Pm, Pp, sm, sp);
            const Real ss = _char_vel_star(rm, rp, um, up, pm, pp, sm, sp);
            assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

            // 4.)
            const Real fr = _hllc_rho(rm, rp, um, up, sm, sp, ss);
            const Real fu = _hllc_pvel(rm, rp, um, up, pm, pp, sm, sp, ss);
            const Real fv = _hllc_vel(rm, rp, vm, vp, um, up, sm, sp, ss);
            const Real fw = _hllc_vel(rm, rp, wm, wp, um, up, sm, sp, ss);
            const Real fe = _hllc_e(rm, rp, um, up, vm, vp, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss);
            const Real fG = _hllc_rho(Gm, Gp, um, up, sm, sp, ss);
            const Real fP = _hllc_rho(Pm, Pp, um, up, sm, sp, ss);
            assert(!isnan(fr)); assert(!isnan(fu)); assert(!isnan(fv)); assert(!isnan(fw)); assert(!isnan(fe)); assert(!isnan(fG)); assert(!isnan(fP));

            const uint_t idx = ID3(iy, ix, iz-3, NY, NXP1);
            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            // 5.)
            xtra_vel[idx] = _extraterm_hllc_vel(um, up, Gm, Gp, Pm, Pp, sm, sp, ss);
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
#endif


#if 0
    if (ix < NXP1 && iy < NY)
    {
        // Process nslices of current chunk
        // iz = 0, 1, 2: left zghost slices
        // iz = nslices+3, nslices+4, nslices+5: right zghost slices
        for (uint_t iz = 3; iz < nslices+3; ++iz)
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
            _xfetch_data(texR, ghostL.r, ghostR.r, ix, iy, iz, global_iz, NXP1, NY, rm3, rm2, rm1, rp1, rp2, rp3);
            assert(rm3 > 0); assert(rm2 > 0); assert(rm1 > 0); assert(rp1 > 0); assert(rp2 > 0); assert(rp3 > 0);

            Real um3, um2, um1, up1, up2, up3;
            _xfetch_data(texU, ghostL.u, ghostR.u, ix, iy, iz, global_iz, NXP1, NY, um3, um2, um1, up1, up2, up3);

            Real vm3, vm2, vm1, vp1, vp2, vp3;
            _xfetch_data(texV, ghostL.v, ghostR.v, ix, iy, iz, global_iz, NXP1, NY, vm3, vm2, vm1, vp1, vp2, vp3);

            Real wm3, wm2, wm1, wp1, wp2, wp3;
            _xfetch_data(texW, ghostL.w, ghostR.w, ix, iy, iz, global_iz, NXP1, NY, wm3, wm2, wm1, wp1, wp2, wp3);

            Real em3, em2, em1, ep1, ep2, ep3;
            _xfetch_data(texE, ghostL.e, ghostR.e, ix, iy, iz, global_iz, NXP1, NY, em3, em2, em1, ep1, ep2, ep3);
            assert(em3 > 0); assert(em2 > 0); assert(em1 > 0); assert(ep1 > 0); assert(ep2 > 0); assert(ep3 > 0);

            Real Gm3, Gm2, Gm1, Gp1, Gp2, Gp3;
            _xfetch_data(texG, ghostL.G, ghostR.G, ix, iy, iz, global_iz, NXP1, NY, Gm3, Gm2, Gm1, Gp1, Gp2, Gp3);
            assert(Gm3 > 0); assert(Gm2 > 0); assert(Gm1 > 0); assert(Gp1 > 0); assert(Gp2 > 0); assert(Gp3 > 0);

            Real Pm3, Pm2, Pm1, Pp1, Pp2, Pp3;
            _xfetch_data(texP, ghostL.P, ghostR.P, ix, iy, iz, global_iz, NXP1, NY, Pm3, Pm2, Pm1, Pp1, Pp2, Pp3);
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
            const Real fr = _hllc_rho(rm, rp, um, up, sm, sp, ss);
            const Real fu = _hllc_pvel(rm, rp, um, up, pm, pp, sm, sp, ss);
            const Real fv = _hllc_vel(rm, rp, vm, vp, um, up, sm, sp, ss);
            const Real fw = _hllc_vel(rm, rp, wm, wp, um, up, sm, sp, ss);
            const Real fe = _hllc_e(rm, rp, um, up, vm, vp, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss);
            const Real fG = _hllc_rho(Gm, Gp, um, up, sm, sp, ss);
            const Real fP = _hllc_rho(Pm, Pp, um, up, sm, sp, ss);
            assert(!isnan(fr)); assert(!isnan(fu)); assert(!isnan(fv)); assert(!isnan(fw)); assert(!isnan(fe)); assert(!isnan(fG)); assert(!isnan(fP));

            const uint_t idx = ID3(ix, iy, iz-3, NXP1, NY);
            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            ///////////////////////////////////////////////////////////////////
            // 5.) RHS for advection equations
            ///////////////////////////////////////////////////////////////////
            xtra_vel[idx] = _extraterm_hllc_vel(um, up, Gm, Gp, Pm, Pp, sm, sp, ss);
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
#endif
}



__global__
void _xflux_left(const uint_t nslices, const uint_t global_iz,
        devPtrSet ghost, devPtrSet flux,
        Real * const xtra_vel,
        Real * const xtra_Gm, Real * const xtra_Gp,
        Real * const xtra_Pm, Real * const xtra_Pp)
{
    const uint_t ix  = blockIdx.x * _NFLUXES_  + threadIdx.x;// actually, threadIdx.x = 0 for all blocks
    const uint_t iy  = blockIdx.y * blockDim.y + threadIdx.y;

    assert(NX >= 8);

    if (ix < NXP1 && iy < NY)
    {
        for (uint_t iz = 3; iz < nslices+3; ++iz) // first and last 3 slices are zghosts
        {
            /* *
             * The general task order is (for each chunk slice along NZ):
             * 1.) Load ghosts from GMEM or tex3D into stencil
             * 2.) Reconstruct primitive values using WENO5/WENO3 (WENO3 = less
             *     registers)
             * 3.) Compute characteristic velocities
             * 4.) Compute fluxes
             * 5.) Compute RHS for advection of G and P
             * */

            // stencils (7 * _STENCIL_WIDTH_ registers per thread)
            Real r[_STENCIL_WIDTH_];
            Real u[_STENCIL_WIDTH_];
            Real v[_STENCIL_WIDTH_];
            Real w[_STENCIL_WIDTH_];
            Real e[_STENCIL_WIDTH_];
            Real G[_STENCIL_WIDTH_];
            Real P[_STENCIL_WIDTH_];

            // 1.)
            if (blockIdx.x == 0)
                _load_halo_stencil_X<0,0,3,_STENCIL_WIDTH_>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghost);
            else
                _load_internal_stencil_X<3>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghost);

            // 2.)
            /* *
             * The reconstructed values are returned in the input. E.g., r[0]
             * and r[1] correspond to rho_minus and rho_plus, respectively for
             * the first face and so on until _NFLUXES_.
             * */
            _reconstruct(r, u, v, w, e, G, P);

            // 3.)
            Real cvel[6], cvel_star[3];
            _char_vel_einfeldt(r[0], r[1], u[0], u[1], e[0], e[1], G[0], G[1], P[0], P[1], cvel[0], cvel[1]);
            _char_vel_einfeldt(r[2], r[3], u[2], u[3], e[2], e[3], G[2], G[3], P[2], P[3], cvel[2], cvel[3]);
            _char_vel_einfeldt(r[4], r[5], u[4], u[5], e[4], e[5], G[4], G[5], P[4], P[5], cvel[4], cvel[5]);
            cvel_star[0] = _char_vel_star(r[0], r[1], u[0], u[1], e[0], e[1], cvel[0], cvel[1]);
            cvel_star[1] = _char_vel_star(r[2], r[3], u[2], u[3], e[2], e[3], cvel[2], cvel[3]);
            cvel_star[2] = _char_vel_star(r[4], r[5], u[4], u[5], e[4], e[5], cvel[4], cvel[5]);

            assert(!isnan(cvel[0])); assert(!isnan(cvel[1])); assert(!isnan(cvel[2]));
            assert(!isnan(cvel[3])); assert(!isnan(cvel[4])); assert(!isnan(cvel[5]));
            assert(!isnan(cvel_star[0])); assert(!isnan(cvel_star[1])); assert(!isnan(cvel_star[2]));

            // 4.)
            const Real fr0 = _hllc_rho(r[0], r[1], u[0], u[1], cvel[0], cvel[1], cvel_star[0]);
            const Real fr1 = _hllc_rho(r[2], r[3], u[2], u[3], cvel[2], cvel[3], cvel_star[1]);
            const Real fr2 = _hllc_rho(r[4], r[5], u[4], u[5], cvel[4], cvel[5], cvel_star[2]);

            const Real fG0 = _hllc_rho(G[0], G[1], u[0], u[1], cvel[0], cvel[1], cvel_star[0]);
            const Real fG1 = _hllc_rho(G[2], G[3], u[2], u[3], cvel[2], cvel[3], cvel_star[1]);
            const Real fG2 = _hllc_rho(G[4], G[5], u[4], u[5], cvel[4], cvel[5], cvel_star[2]);

            const Real fP0 = _hllc_rho(P[0], P[1], u[0], u[1], cvel[0], cvel[1], cvel_star[0]);
            const Real fP1 = _hllc_rho(P[2], P[3], u[2], u[3], cvel[2], cvel[3], cvel_star[1]);
            const Real fP2 = _hllc_rho(P[4], P[5], u[4], u[5], cvel[4], cvel[5], cvel_star[2]);

            const Real fe0 = _hllc_e(r[0], r[1], u[0], u[1], v[0], v[1], w[0], w[1], e[0], e[1], G[0], G[1], P[0], P[1], cvel[0], cvel[1], cvel_star[0]);
            const Real fe1 = _hllc_e(r[2], r[3], u[2], u[3], v[2], v[3], w[2], w[3], e[2], e[3], G[2], G[3], P[2], P[3], cvel[2], cvel[3], cvel_star[1]);
            const Real fe2 = _hllc_e(r[4], r[5], u[4], u[5], v[4], v[5], w[4], w[5], e[4], e[5], G[4], G[5], P[4], P[5], cvel[4], cvel[5], cvel_star[2]);

            const Real fu0 = _hllc_pvel(r[0], r[1], u[0], u[1], e[0], e[1], cvel[0], cvel[1], cvel_star[0]);
            const Real fu1 = _hllc_pvel(r[2], r[3], u[2], u[3], e[2], e[3], cvel[2], cvel[3], cvel_star[1]);
            const Real fu2 = _hllc_pvel(r[4], r[5], u[4], u[5], e[4], e[5], cvel[4], cvel[5], cvel_star[2]);

            const Real fv0 = _hllc_vel(r[0], r[1], v[0], v[1], u[0], u[1], cvel[0], cvel[1], cvel_star[0]);
            const Real fv1 = _hllc_vel(r[2], r[3], v[2], v[3], u[2], u[3], cvel[2], cvel[3], cvel_star[1]);
            const Real fv2 = _hllc_vel(r[4], r[5], v[4], v[5], u[4], u[5], cvel[4], cvel[5], cvel_star[2]);

            const Real fw0 = _hllc_vel(r[0], r[1], w[0], w[1], u[0], u[1], cvel[0], cvel[1], cvel_star[0]);
            const Real fw1 = _hllc_vel(r[2], r[3], w[2], w[3], u[2], u[3], cvel[2], cvel[3], cvel_star[1]);
            const Real fw2 = _hllc_vel(r[4], r[5], w[4], w[5], u[4], u[5], cvel[4], cvel[5], cvel_star[2]);

            const Real xtra_vel0 = _extraterm_hllc_vel(u[0], u[1], G[0], G[1], P[0], P[1], cvel[0], cvel[1], cvel_star[0]);
            const Real xtra_vel1 = _extraterm_hllc_vel(u[2], u[3], G[2], G[3], P[2], P[3], cvel[2], cvel[3], cvel_star[1]);
            const Real xtra_vel2 = _extraterm_hllc_vel(u[4], u[5], G[4], G[5], P[4], P[5], cvel[4], cvel[5], cvel_star[2]);

            const uint_t idx0 = ID3(iy, ix+0, iz-3, NY, NXP1);
            const uint_t idx1 = ID3(iy, ix+1, iz-3, NY, NXP1);
            const uint_t idx2 = ID3(iy, ix+2, iz-3, NY, NXP1);
            flux.r[idx0] = fr0; flux.r[idx1] = fr1; flux.r[idx2] = fr2;
            flux.u[idx0] = fu0; flux.u[idx1] = fu1; flux.u[idx2] = fu2;
            flux.v[idx0] = fv0; flux.v[idx1] = fv1; flux.v[idx2] = fv2;
            flux.w[idx0] = fw0; flux.w[idx1] = fw1; flux.w[idx2] = fw2;
            flux.e[idx0] = fe0; flux.e[idx1] = fe1; flux.e[idx2] = fe2;
            flux.G[idx0] = fG0; flux.G[idx1] = fG1; flux.G[idx2] = fG2;
            flux.P[idx0] = fP0; flux.P[idx1] = fP1; flux.P[idx2] = fP2;

            // 5.)
            xtra_vel[idx0] = xtra_vel0; xtra_vel[idx1] = xtra_vel1; xtra_vel[idx2] = xtra_vel2;
            xtra_Gm [idx0] = G[0];      xtra_Gm [idx1] = G[2];      xtra_Gm [idx2] = G[4];
            xtra_Gp [idx0] = G[1];      xtra_Gp [idx1] = G[3];      xtra_Gp [idx2] = G[5];
            xtra_Pm [idx0] = P[0];      xtra_Pm [idx1] = P[2];      xtra_Pm [idx2] = P[4];
            xtra_Pp [idx0] = P[1];      xtra_Pp [idx1] = P[3];      xtra_Pp [idx2] = P[5];
        }
    }
}

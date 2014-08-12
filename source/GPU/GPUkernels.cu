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
//                             DEVICE FUNCTIONS                              //
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
//                                  KERNELS                                  //
///////////////////////////////////////////////////////////////////////////////
#define _NFLUXES_ 3
#define _STENCIL_WIDTH_ 8 // 6 + _NFLUXES_ - 1

__global__
void _xextraterm_hllc(const uint_t nslices,
        const Real * const __restrict__ Gm, const Real * const __restrict__ Gp,
        const Real * const __restrict__ Pm, const Real * const __restrict__ Pp,
        const Real * const __restrict__ vel,
        Real * const __restrict__ sumG, Real * const __restrict__ sumP, Real * const __restrict__ divU)
{
    const uint_t ix = blockIdx.x * _TILE_DIM_ + threadIdx.x;
    const uint_t iy = blockIdx.y * _TILE_DIM_ + threadIdx.y;

    // limiting resource, but runs faster by using 3 buffers
    __shared__ Real smem1[_TILE_DIM_][_TILE_DIM_+1];
    __shared__ Real smem2[_TILE_DIM_][_TILE_DIM_+1];
    __shared__ Real smem3[_TILE_DIM_][_TILE_DIM_+1];

    if (ix < NX && iy < NY)
    {
        // transpose
        const uint_t iyT = blockIdx.y * _TILE_DIM_ + threadIdx.x;
        const uint_t ixT = blockIdx.x * _TILE_DIM_ + threadIdx.y;

        // per thread:
        // LOADS  = nslices * (6 * _TILE_DIM_/_BLOCK_ROWS_)
        // STORES = nslices * (3 * _TILE_DIM_/_BLOCK_ROWS_)
        // total words transferred per thread:
        // WORDS  = nslices * (9 * _TILE_DIM_/_BLOCK_ROWS_)
        for (uint_t iz = 0; iz < nslices; ++iz)
        {
            for (int i = 0; i < _TILE_DIM_; i += _BLOCK_ROWS_)
            {
                smem1[threadIdx.x][threadIdx.y+i] = Gp[ID3(iyT,ixT+i,iz,NY,NXP1)]      + Gm[ID3(iyT,(ixT+1)+i,iz,NY,NXP1)];
                smem2[threadIdx.x][threadIdx.y+i] = Pp[ID3(iyT,ixT+i,iz,NY,NXP1)]      + Pm[ID3(iyT,(ixT+1)+i,iz,NY,NXP1)];
                smem3[threadIdx.x][threadIdx.y+i] = vel[ID3(iyT,(ixT+1)+i,iz,NY,NXP1)] - vel[ID3(iyT,ixT+i,iz,NY,NXP1)];
            }
            __syncthreads();
            for (int i = 0; i < _TILE_DIM_; i += _BLOCK_ROWS_)
            {
                sumG[ID3(ix,iy+i,iz,NX,NY)] = smem1[threadIdx.y+i][threadIdx.x];
                sumP[ID3(ix,iy+i,iz,NX,NY)] = smem2[threadIdx.y+i][threadIdx.x];
                divU[ID3(ix,iy+i,iz,NX,NY)] = smem3[threadIdx.y+i][threadIdx.x];
            }
            __syncthreads();
        }
    }
}


__global__
void _yextraterm_hllc(const uint_t nslices,
        const Real * const __restrict__ Gm, const Real * const __restrict__ Gp,
        const Real * const __restrict__ Pm, const Real * const __restrict__ Pp,
        const Real * const __restrict__ vel,
        Real * const __restrict__ sumG, Real * const __restrict__ sumP, Real * const __restrict__ divU)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        // per thread:
        // LOADS  = nslices * 9
        // STORES = nslices * 3
        // total words transferred per thread:
        // WORDS  = nslices * 12
        for (uint_t iz = 0; iz < nslices; ++iz)
        {
            const uint_t idx  = ID3(ix,iy,iz,NX,NY);
            const uint_t idxm = ID3(ix,iy,iz,NX,NYP1);
            const uint_t idxp = ID3(ix,(iy+1),iz,NX,NYP1);
            Real tq = Gp[idxm];
            Real tr = Pp[idxm];
            Real ts = vel[idxp];
            tq = tq + Gm[idxp];
            tr = tr + Pm[idxp];
            ts = ts - vel[idxm];
            sumG[idx] += tq;
            sumP[idx] += tr;
            divU[idx] += ts;
        }
    }
}


__global__
void _zextraterm_hllc(const uint_t nslices,
        const Real * const __restrict__ Gm, const Real * const __restrict__ Gp,
        const Real * const __restrict__ Pm, const Real * const __restrict__ Pp,
        const Real * const __restrict__ vel,
        Real * const __restrict__ sumG, Real * const __restrict__ sumP, Real * const __restrict__ divU)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        // per thread:
        // LOADS  = nslices * 9
        // STORES = nslices * 3
        // total words transferred per thread:
        // WORDS  = nslices * 12
        for (uint_t iz = 0; iz < nslices; ++iz)
        {
            const uint_t idx  = ID3(ix,iy,iz,NX,NY);
            const uint_t idxm = ID3(ix,iy,iz,NX,NY);
            const uint_t idxp = ID3(ix,iy,(iz+1),NX,NY);
            Real tq = Gp[idxm];
            Real tr = Pp[idxm];
            Real ts = vel[idxp];
            tq = tq + Gm[idxp];
            tr = tr + Pm[idxp];
            ts = ts - vel[idxm];
            sumG[idx] += tq;
            sumP[idx] += tr;
            divU[idx] += ts;
        }
    }
}


__global__
void _xflux(const uint_t nslices, const uint_t global_iz,
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
                _load_3X<0,0,3,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(3*GMEM + 3*TEX)
            else if (1 == ix)
                _load_2X<0,0,2,1>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(2*GMEM + 4*TEX)
            else if (2 == ix)
                _load_1X<0,0,1,2>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(1*GMEM + 5*TEX)
            else if (NXP1-3 == ix)
                _load_1X<NXP1-6,5,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NXP1-2 == ix)
                _load_2X<NXP1-5,4,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NXP1-1 == ix)
                _load_3X<NXP1-4,3,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else
                _load_internal_X(ix, iy, iz, r, u, v, w, e, G, P, global_iz, NULL); // load 7*(6*TEX)

            // 2.)
            // convert to primitive variables
#pragma unroll 6
            for (uint_t i = 0; i < 6; ++i)
            {
                e[i] = (e[i] - 0.5f*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i])/r[i] - P[i]) / G[i];
                u[i] = u[i]/r[i];
                v[i] = v[i]/r[i];
                w[i] = w[i]/r[i];
            } // 6 x (8 MUL/ADD/SUB + 5 DIV) = 78 FLOPS

///////////////////////////////////////////////////////////////////////////////
            // TEST MORE SOFTWARE PIPELINE //
            /* Real rm, rp, um, up, vm, vp, wm, wp, pm, pp, Gm, Gp, Pm, Pp; */
            /* _weno_reconstruction(rm, rp, r); */
            /* _weno_reconstruction(Gm, Gp, G); */
            /* _weno_reconstruction(Pm, Pp, P); */
            /* _weno_reconstruction(pm, pp, e); */
            /* _weno_reconstruction(um, up, u); */
            /* _weno_reconstruction(vm, vp, v); */
            /* _weno_reconstruction(wm, wp, w); */
            /* rm = _weno_clip_minus(rm, r[1], r[2], r[3]); */
            /* rp = _weno_clip_pluss(rp, r[2], r[3], r[4]); */
            /* Gm = _weno_clip_minus(Gm, G[1], G[2], G[3]); */
            /* Gp = _weno_clip_pluss(Gp, G[2], G[3], G[4]); */
            /* Pm = _weno_clip_minus(Pm, P[1], P[2], P[3]); */
            /* Pp = _weno_clip_pluss(Pp, P[2], P[3], P[4]); */
            /* pm = _weno_clip_minus(pm, e[1], e[2], e[3]); */
            /* pp = _weno_clip_pluss(pp, e[2], e[3], e[4]); */
            /* um = _weno_clip_minus(um, u[1], u[2], u[3]); */
            /* up = _weno_clip_pluss(up, u[2], u[3], u[4]); */
            /* vm = _weno_clip_minus(vm, v[1], v[2], v[3]); */
            /* vp = _weno_clip_pluss(vp, v[2], v[3], v[4]); */
            /* wm = _weno_clip_minus(wm, w[1], w[2], w[3]); */
            /* wp = _weno_clip_pluss(wp, w[2], w[3], w[4]); */
            /* assert(!isnan(rp)); assert(!isnan(rm)); */
            /* assert(!isnan(Gp)); assert(!isnan(Gm)); */
            /* assert(!isnan(Pp)); assert(!isnan(Pm)); */
            /* assert(!isnan(pp)); assert(!isnan(pm)); */
            /* assert(!isnan(up)); assert(!isnan(um)); */
            /* assert(!isnan(vp)); assert(!isnan(vm)); */
            /* assert(!isnan(wp)); assert(!isnan(wm)); */
///////////////////////////////////////////////////////////////////////////////

            const Real rm = _weno_minus_clipped(r[0], r[1], r[2], r[3], r[4]); // 96 FLOP (6 DIV)
            const Real rp = _weno_pluss_clipped(r[1], r[2], r[3], r[4], r[5]); // 96 FLOP (6 DIV)
            assert(!isnan(rp)); assert(!isnan(rm));

            const Real Gm = _weno_minus_clipped(G[0], G[1], G[2], G[3], G[4]); // 96 FLOP (6 DIV)
            const Real Gp = _weno_pluss_clipped(G[1], G[2], G[3], G[4], G[5]); // 96 FLOP (6 DIV)
            assert(!isnan(Gp)); assert(!isnan(Gm));

            const Real Pm = _weno_minus_clipped(P[0], P[1], P[2], P[3], P[4]); // 96 FLOP (6 DIV)
            const Real Pp = _weno_pluss_clipped(P[1], P[2], P[3], P[4], P[5]); // 96 FLOP (6 DIV)
            assert(!isnan(Pp)); assert(!isnan(Pm));

            const Real pm = _weno_minus_clipped(e[0], e[1], e[2], e[3], e[4]); // 96 FLOP (6 DIV)
            const Real pp = _weno_pluss_clipped(e[1], e[2], e[3], e[4], e[5]); // 96 FLOP (6 DIV)
            assert(!isnan(pp)); assert(!isnan(pm));

            const Real um = _weno_minus_clipped(u[0], u[1], u[2], u[3], u[4]); // 96 FLOP (6 DIV)
            const Real up = _weno_pluss_clipped(u[1], u[2], u[3], u[4], u[5]); // 96 FLOP (6 DIV)
            assert(!isnan(up)); assert(!isnan(um));

            const Real vm = _weno_minus_clipped(v[0], v[1], v[2], v[3], v[4]); // 96 FLOP (6 DIV)
            const Real vp = _weno_pluss_clipped(v[1], v[2], v[3], v[4], v[5]); // 96 FLOP (6 DIV)
            assert(!isnan(vp)); assert(!isnan(vm));

            const Real wm = _weno_minus_clipped(w[0], w[1], w[2], w[3], w[4]); // 96 FLOP (6 DIV)
            const Real wp = _weno_pluss_clipped(w[1], w[2], w[3], w[4], w[5]); // 96 FLOP (6 DIV)
            assert(!isnan(wp)); assert(!isnan(wm));

            // 3.)
            Real sm, sp;
            _char_vel_einfeldt(rm, rp, um, up, pm, pp, Gm, Gp, Pm, Pp, sm, sp); // 29 FLOP (6 DIV)
            const Real ss = _char_vel_star(rm, rp, um, up, pm, pp, sm, sp); // 11 FLOP (1 DIV)
            assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

            // 4.)
            const Real fr = _hllc_rho(rm, rp, um, up, sm, sp, ss); // 23 FLOP (2 DIV)
            const Real fu = _hllc_pvel(rm, rp, um, up, pm, pp, sm, sp, ss); // 29 FLOP (2 DIV)
            const Real fv = _hllc_vel(rm, rp, vm, vp, um, up, sm, sp, ss); // 25 FLOP (2 DIV)
            const Real fw = _hllc_vel(rm, rp, wm, wp, um, up, sm, sp, ss); // 25 FLOP (2 DIV)
            const Real fe = _hllc_e(rm, rp, um, up, vm, vp, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss); // 59 FLOP (4 DIV)
            const Real fG = _hllc_rho(Gm, Gp, um, up, sm, sp, ss); // 23 FLOP (2 DIV)
            const Real fP = _hllc_rho(Pm, Pp, um, up, sm, sp, ss); // 23 FLOP (2 DIV)
            assert(!isnan(fr)); assert(!isnan(fu)); assert(!isnan(fv)); assert(!isnan(fw)); assert(!isnan(fe)); assert(!isnan(fG)); assert(!isnan(fP));

            const Real hllc_vel = _extraterm_hllc_vel(um, up, Gm, Gp, Pm, Pp, sm, sp, ss); // 19 FLOP (2 DIV)

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
void _yflux(const uint_t nslices, const uint_t global_iz,
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
                _load_3Y<0,0,3,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(3*GMEM + 3*TEX)
            else if (1 == iy)
                _load_2Y<0,0,2,1>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(2*GMEM + 4*TEX)
            else if (2 == iy)
                _load_1Y<0,0,1,2>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL); // load 7*(1*GMEM + 5*TEX)
            else if (NYP1-3 == iy)
                _load_1Y<NYP1-6,5,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NYP1-2 == iy)
                _load_2Y<NYP1-5,4,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NYP1-1 == iy)
                _load_3Y<NYP1-4,3,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else
                _load_internal_Y(ix, iy, iz, r, u, v, w, e, G, P, global_iz, NULL); // load 7*(6*TEX)

            // 2.)
            // convert to primitive variables
#pragma unroll 6
            for (uint_t i = 0; i < 6; ++i)
            {
                e[i] = (e[i] - 0.5f*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i])/r[i] - P[i]) / G[i];
                u[i] = u[i]/r[i];
                v[i] = v[i]/r[i];
                w[i] = w[i]/r[i];
            } // 6 x (8 MUL/ADD/SUB + 5 DIV) = 78 FLOPS

            const Real rm = _weno_minus_clipped(r[0], r[1], r[2], r[3], r[4]); // 96 FLOP (6 DIV)
            const Real rp = _weno_pluss_clipped(r[1], r[2], r[3], r[4], r[5]); // 96 FLOP (6 DIV)
            assert(!isnan(rp)); assert(!isnan(rm));

            const Real Gm = _weno_minus_clipped(G[0], G[1], G[2], G[3], G[4]); // 96 FLOP (6 DIV)
            const Real Gp = _weno_pluss_clipped(G[1], G[2], G[3], G[4], G[5]); // 96 FLOP (6 DIV)
            assert(!isnan(Gp)); assert(!isnan(Gm));

            const Real Pm = _weno_minus_clipped(P[0], P[1], P[2], P[3], P[4]); // 96 FLOP (6 DIV)
            const Real Pp = _weno_pluss_clipped(P[1], P[2], P[3], P[4], P[5]); // 96 FLOP (6 DIV)
            assert(!isnan(Pp)); assert(!isnan(Pm));

            const Real pm = _weno_minus_clipped(e[0], e[1], e[2], e[3], e[4]); // 96 FLOP (6 DIV)
            const Real pp = _weno_pluss_clipped(e[1], e[2], e[3], e[4], e[5]); // 96 FLOP (6 DIV)
            assert(!isnan(pp)); assert(!isnan(pm));

            const Real vm = _weno_minus_clipped(v[0], v[1], v[2], v[3], v[4]); // 96 FLOP (6 DIV)
            const Real vp = _weno_pluss_clipped(v[1], v[2], v[3], v[4], v[5]); // 96 FLOP (6 DIV)
            assert(!isnan(vp)); assert(!isnan(vm));

            const Real um = _weno_minus_clipped(u[0], u[1], u[2], u[3], u[4]); // 96 FLOP (6 DIV)
            const Real up = _weno_pluss_clipped(u[1], u[2], u[3], u[4], u[5]); // 96 FLOP (6 DIV)
            assert(!isnan(up)); assert(!isnan(um));

            const Real wm = _weno_minus_clipped(w[0], w[1], w[2], w[3], w[4]); // 96 FLOP (6 DIV)
            const Real wp = _weno_pluss_clipped(w[1], w[2], w[3], w[4], w[5]); // 96 FLOP (6 DIV)
            assert(!isnan(wp)); assert(!isnan(wm));

            // 3.)
            Real sm, sp;
            _char_vel_einfeldt(rm, rp, vm, vp, pm, pp, Gm, Gp, Pm, Pp, sm, sp); // 29 FLOP (6 DIV)
            const Real ss = _char_vel_star(rm, rp, vm, vp, pm, pp, sm, sp); // 11 FLOP (1 DIV)
            assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

            // 4.)
            const Real fr = _hllc_rho(rm, rp, vm, vp, sm, sp, ss); // 23 FLOP (2 DIV)
            const Real fu = _hllc_vel(rm, rp, um, up, vm, vp, sm, sp, ss); // 25 FLOP (2 DIV)
            const Real fv = _hllc_pvel(rm, rp, vm, vp, pm, pp, sm, sp, ss); // 29 FLOP (2 DIV)
            const Real fw = _hllc_vel(rm, rp, wm, wp, vm, vp, sm, sp, ss); // 25 FLOP (2 DIV)
            const Real fe = _hllc_e(rm, rp, vm, vp, um, up, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss); // 59 FLOP (4 DIV)
            const Real fG = _hllc_rho(Gm, Gp, vm, vp, sm, sp, ss); // 23 FLOP (2 DIV)
            const Real fP = _hllc_rho(Pm, Pp, vm, vp, sm, sp, ss); // 23 FLOP (2 DIV)
            assert(!isnan(fr)); assert(!isnan(fu)); assert(!isnan(fv)); assert(!isnan(fw)); assert(!isnan(fe)); assert(!isnan(fG)); assert(!isnan(fP));

            const Real hllc_vel = _extraterm_hllc_vel(vm, vp, Gm, Gp, Pm, Pp, sm, sp, ss); // 19 FLOP (2 DIV)

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
void _zflux(const uint_t nslices, DevicePointer flux,
        Real * const __restrict__ xtra_vel,
        Real * const __restrict__ xtra_Gm, Real * const __restrict__ xtra_Gp,
        Real * const __restrict__ xtra_Pm, Real * const __restrict__ xtra_Pp,
        const uint_t global_iz = 0)
{
    /* *
     * Notes:
     * ======
     * 1.) NX = NodeBlock::sizeX
     * 2.) NY = NodeBlock::sizeY
     * 3.) NZ = NodeBlock::sizeZ
     * 4.) nslices = number of slices for currently processed chunk
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
            _load_internal_Z(ix, iy, iz, r, u, v, w, e, G, P, 0, NULL); // load 7*(6*TEX)

            // 2.)
            // convert to primitive variables
#pragma unroll 6
            for (uint_t i = 0; i < 6; ++i)
            {
                e[i] = (e[i] - 0.5f*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i])/r[i] - P[i]) / G[i];
                u[i] = u[i]/r[i];
                v[i] = v[i]/r[i];
                w[i] = w[i]/r[i];
            } // 6 x (8 MUL/ADD/SUB + 5 DIV) = 78 FLOPS

            const Real rm = _weno_minus_clipped(r[0], r[1], r[2], r[3], r[4]); // 96 FLOP (6 DIV)
            const Real rp = _weno_pluss_clipped(r[1], r[2], r[3], r[4], r[5]); // 96 FLOP (6 DIV)
            assert(!isnan(rp)); assert(!isnan(rm));

            const Real Gm = _weno_minus_clipped(G[0], G[1], G[2], G[3], G[4]); // 96 FLOP (6 DIV)
            const Real Gp = _weno_pluss_clipped(G[1], G[2], G[3], G[4], G[5]); // 96 FLOP (6 DIV)
            assert(!isnan(Gp)); assert(!isnan(Gm));

            const Real Pm = _weno_minus_clipped(P[0], P[1], P[2], P[3], P[4]); // 96 FLOP (6 DIV)
            const Real Pp = _weno_pluss_clipped(P[1], P[2], P[3], P[4], P[5]); // 96 FLOP (6 DIV)
            assert(!isnan(Pp)); assert(!isnan(Pm));

            const Real pm = _weno_minus_clipped(e[0], e[1], e[2], e[3], e[4]); // 96 FLOP (6 DIV)
            const Real pp = _weno_pluss_clipped(e[1], e[2], e[3], e[4], e[5]); // 96 FLOP (6 DIV)
            assert(!isnan(pp)); assert(!isnan(pm));

            const Real wm = _weno_minus_clipped(w[0], w[1], w[2], w[3], w[4]); // 96 FLOP (6 DIV)
            const Real wp = _weno_pluss_clipped(w[1], w[2], w[3], w[4], w[5]); // 96 FLOP (6 DIV)
            assert(!isnan(wp)); assert(!isnan(wm));

            const Real um = _weno_minus_clipped(u[0], u[1], u[2], u[3], u[4]); // 96 FLOP (6 DIV)
            const Real up = _weno_pluss_clipped(u[1], u[2], u[3], u[4], u[5]); // 96 FLOP (6 DIV)
            assert(!isnan(up)); assert(!isnan(um));

            const Real vm = _weno_minus_clipped(v[0], v[1], v[2], v[3], v[4]); // 96 FLOP (6 DIV)
            const Real vp = _weno_pluss_clipped(v[1], v[2], v[3], v[4], v[5]); // 96 FLOP (6 DIV)
            assert(!isnan(vp)); assert(!isnan(vm));

            // 3.)
            Real sm, sp;
            _char_vel_einfeldt(rm, rp, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp); // 29 FLOP (6 DIV)
            const Real ss = _char_vel_star(rm, rp, wm, wp, pm, pp, sm, sp); // 11 FLOP (1 DIV)
            assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

            // 4.)
            const Real fr = _hllc_rho(rm, rp, wm, wp, sm, sp, ss); // 23 FLOP (2 DIV)
            const Real fu = _hllc_vel(rm, rp, um, up, wm, wp, sm, sp, ss); // 25 FLOP (2 DIV)
            const Real fv = _hllc_vel(rm, rp, vm, vp, wm, wp, sm, sp, ss); // 25 FLOP (2 DIV)
            const Real fw = _hllc_pvel(rm, rp, wm, wp, pm, pp, sm, sp, ss); // 29 FLOP (2 DIV)
            const Real fe = _hllc_e(rm, rp, wm, wp, um, up, vm, vp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss); // 59 FLOP (4 DIV)
            const Real fG = _hllc_rho(Gm, Gp, wm, wp, sm, sp, ss); // 23 FLOP (2 DIV)
            const Real fP = _hllc_rho(Pm, Pp, wm, wp, sm, sp, ss); // 23 FLOP (2 DIV)

            const Real hllc_vel = _extraterm_hllc_vel(wm, wp, Gm, Gp, Pm, Pp, sm, sp, ss); // 19 FLOP (2 DIV)

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
void _divergence(const uint_t nslices,
        const DevicePointer xflux, const DevicePointer yflux, const DevicePointer zflux,
        DevicePointer rhs, const Real a, const Real dtinvh, const DevicePointer tmp,
        const Real * const sumG, const Real * const sumP, const Real * const divU)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        Real fxp, fxm, fyp, fym, fzp, fzm;
        const Real factor6 = 1.0f / 6.0f;

        for (uint_t iz = 0; iz < nslices; ++iz)
        {
            const uint_t idx = ID3(ix, iy, iz, NX, NY);

            _fetch_flux(ix, iy, iz, xflux.r, yflux.r, zflux.r, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_r = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.r[idx] = a*tmp.r[idx] - rhs_r;

            _fetch_flux(ix, iy, iz, xflux.u, yflux.u, zflux.u, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_u = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.u[idx] = a*tmp.u[idx] - rhs_u;

            _fetch_flux(ix, iy, iz, xflux.v, yflux.v, zflux.v, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_v = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.v[idx] = a*tmp.v[idx] - rhs_v;

            _fetch_flux(ix, iy, iz, xflux.w, yflux.w, zflux.w, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_w = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.w[idx] = a*tmp.w[idx] - rhs_w;

            _fetch_flux(ix, iy, iz, xflux.e, yflux.e, zflux.e, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_e = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm);
            rhs.e[idx] = a*tmp.e[idx] - rhs_e;

            _fetch_flux(ix, iy, iz, xflux.G, yflux.G, zflux.G, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_G = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm   - divU[idx] * sumG[idx] * factor6);
            rhs.G[idx] = a*tmp.G[idx] - rhs_G;

            _fetch_flux(ix, iy, iz, xflux.P, yflux.P, zflux.P, fxp, fxm, fyp, fym, fzp, fzm);
            const Real rhs_P = dtinvh*(fxp - fxm + fyp - fym + fzp - fzm   - divU[idx] * sumP[idx] * factor6);
            rhs.P[idx] = a*tmp.P[idx] - rhs_P;
        }
    }
}


__global__
void _update(const uint_t nslices, const Real b, DevicePointer tmp, const DevicePointer rhs)
{
    const uint_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < NX && iy < NY)
    {
        for (uint_t iz = 0; iz < nslices; ++iz)
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
            const Real r = tex3D(texR, ix, iy, iz);
            const Real u = tex3D(texU, ix, iy, iz);
            const Real v = tex3D(texV, ix, iy, iz);
            const Real w = tex3D(texW, ix, iy, iz);
            const Real e = tex3D(texE, ix, iy, iz);
            const Real G = tex3D(texG, ix, iy, iz);
            const Real P = tex3D(texP, ix, iy, iz);

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
            assert(sos > 0);
            atomicMax(g_maxSOS, __float_as_int(sos));
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
//                              KERNEL WRAPPERS                              //
///////////////////////////////////////////////////////////////////////////////
void GPU::xflux(const uint_t nslices, const uint_t global_iz)
{
#ifndef _MUTE_GPU_
    DevicePointer xghostL(d_xgl);
    DevicePointer xghostR(d_xgr);
    DevicePointer xflux(d_xflux);

    {
        const dim3 blocks(1, _NTHREADS_, 1);
        const dim3 grid(NXP1, (NY + _NTHREADS_ -1)/_NTHREADS_, 1);
        GPU::profiler.push_startCUDA("_XFLUX", &stream1);
        _xflux<<<grid, blocks, 0, stream1>>>(nslices, global_iz, xghostL, xghostR, xflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
        GPU::profiler.pop_stopCUDA();
    }

    {
        const dim3 xtraBlocks(_TILE_DIM_, _BLOCK_ROWS_, 1);
        const dim3 xtraGrid((NX + _TILE_DIM_ - 1)/_TILE_DIM_, (NY + _TILE_DIM_ - 1)/_TILE_DIM_, 1);
        GPU::profiler.push_startCUDA("_XEXTRATERM", &stream1);
        _xextraterm_hllc<<<xtraGrid, xtraBlocks, 0, stream1>>>(nslices, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
        GPU::profiler.pop_stopCUDA();

        // 70.1% of Peak BW (w/ ECC) on K20c
        const uint_t PTS_PER_SLICE = NX * NY;
        const uint_t total_words = PTS_PER_SLICE * (nslices * 9);
        /* printf("XEXTRA GB = %f\n", total_words*4./1024./1024./1024.); */
    }
#endif
}


void GPU::yflux(const uint_t nslices, const uint_t global_iz)
{
#ifndef _MUTE_GPU_
    DevicePointer yghostL(d_ygl);
    DevicePointer yghostR(d_ygr);
    DevicePointer yflux(d_yflux);

    const dim3 blocks(_NTHREADS_, 1, 1);

    {
        const dim3 grid((NX + _NTHREADS_ -1) / _NTHREADS_, NYP1, 1);
        GPU::profiler.push_startCUDA("_YFLUX", &stream1);
        _yflux<<<grid, blocks, 0, stream1>>>(nslices, global_iz, yghostL, yghostR, yflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
        GPU::profiler.pop_stopCUDA();
    }

    {
        const dim3 grid((NX + _NTHREADS_ -1) / _NTHREADS_, NY, 1);
        GPU::profiler.push_startCUDA("_YEXTRATERM", &stream1);
        _yextraterm_hllc<<<grid, blocks, 0, stream1>>>(nslices, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
        GPU::profiler.pop_stopCUDA();

        // 78.6% of Peak BW (w/ ECC) on K20c
        const uint_t PTS_PER_SLICE = NX * NY;
        const uint_t total_words = PTS_PER_SLICE * (nslices * 12);
        /* printf("YEXTRA GB = %f\n", total_words*4./1024./1024./1024.); */
    }
#endif
}


void GPU::zflux(const uint_t nslices)
{
#ifndef _MUTE_GPU_
    DevicePointer zflux(d_zflux);

    const dim3 grid((NX + _NTHREADS_ -1) / _NTHREADS_, NY, 1);
    const dim3 blocks(_NTHREADS_, 1, 1);

    GPU::profiler.push_startCUDA("_ZFLUX", &stream1);
    _zflux<<<grid, blocks, 0, stream1>>>(nslices, zflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
    GPU::profiler.pop_stopCUDA();

    GPU::profiler.push_startCUDA("_ZEXTRATERM", &stream1);
    _zextraterm_hllc<<<grid, blocks, 0, stream1>>>(nslices, d_Gm, d_Gp, d_Pm, d_Pp, d_hllc_vel, d_sumG, d_sumP, d_divU);
    GPU::profiler.pop_stopCUDA();

    // 76.7% of Peak BW (w/ ECC) on K20c
    const uint_t PTS_PER_SLICE = NX * NY;
    const uint_t total_words = PTS_PER_SLICE * (nslices * 12);
    /* printf("ZEXTRA GB = %f\n", total_words*4./1024./1024./1024.); */
#endif
}


void GPU::divergence(const Real a, const Real dtinvh, const uint_t nslices)
{
#ifndef _MUTE_GPU_
    cudaStreamWaitEvent(stream1, h2d_tmp_completed, 0);

    DevicePointer xflux(d_xflux);
    DevicePointer yflux(d_yflux);
    DevicePointer zflux(d_zflux);
    DevicePointer rhs(d_rhs);
    DevicePointer tmp(d_tmp);

    const dim3 grid((NX + _NTHREADS_ -1) / _NTHREADS_, NY, 1);
    const dim3 blocks(_NTHREADS_, 1, 1);

    GPU::profiler.push_startCUDA("_DIVERGENCE", &stream1);
    _divergence<<<grid, blocks, 0, stream1>>>(nslices, xflux, yflux, zflux, rhs, a, dtinvh, tmp, d_sumG, d_sumP, d_divU);
    GPU::profiler.pop_stopCUDA();

    cudaEventRecord(divergence_completed, stream1);
#endif
}


void GPU::update(const Real b, const uint_t nslices)
{
#ifndef _MUTE_GPU_
    DevicePointer tmp(d_tmp);
    DevicePointer rhs(d_rhs);

    const dim3 grid((NX + _NTHREADS_ -1) / _NTHREADS_, NY, 1);
    const dim3 blocks(_NTHREADS_, 1, 1);

    GPU::profiler.push_startCUDA("_UPDATE", &stream1);
    _update<<<grid, blocks, 0, stream1>>>(nslices, b, tmp, rhs);
    GPU::profiler.pop_stopCUDA();

    cudaEventRecord(update_completed, stream1);
#endif
}


void GPU::MaxSpeedOfSound(const uint_t nslices)
{
#ifndef _MUTE_GPU_
    const dim3 grid((NX + _NTHREADS_ -1) / _NTHREADS_, NY, 1);
    const dim3 blocks(_NTHREADS_, 1, 1);

    GPU::profiler.push_startCUDA("_MAXSOS", &stream1);
    _maxSOS<<<grid, blocks, 0, stream1>>>(nslices, d_maxSOS);
    GPU::profiler.pop_stopCUDA();
#endif
}

///////////////////////////////////////////////////////////////////////////
// TEST SECTION
///////////////////////////////////////////////////////////////////////////
void GPU::TestKernel()
{
    DevicePointer xghostL(d_xgl);
    DevicePointer xghostR(d_xgr);
    DevicePointer xflux(d_xflux);

    DevicePointer yghostL(d_ygl);
    DevicePointer yghostR(d_ygr);
    DevicePointer yflux(d_yflux);

    DevicePointer zflux(d_zflux);


    // rearrange GPU memory for TEST
    for (int var = 0; var < 7; ++var)
    {
        cudaFree(d_tmp[var]);
        cudaFree(d_rhs[var]);
    }
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

        cudaStream_t *_s = (cudaStream_t *) malloc(3*sizeof(cudaStream_t));
        for (int i = 0; i < 3; ++i)
            cudaStreamCreate(&(_s[i]));

        GPU::profiler.push_startCUDA("_XFLUX", &_s[0]);
        _xflux<<<xgrid, xblocks, 0, _s[0]>>>(nslices, 0, xghostL, xghostR, xflux, d_extra_X[0], d_extra_X[1], d_extra_X[2], d_extra_X[3], d_extra_X[4]);
        GPU::profiler.pop_stopCUDA();

        GPU::profiler.push_startCUDA("_YFLUX", &_s[0]);
        _yflux<<<ygrid, yblocks, 0, _s[0]>>>(nslices, 0, yghostL, yghostR, yflux, d_extra_Y[0], d_extra_Y[1], d_extra_Y[2], d_extra_Y[3], d_extra_Y[4]);
        GPU::profiler.pop_stopCUDA();

        GPU::profiler.push_startCUDA("_ZFLUX", &_s[0]);
        _zflux<<<zgrid, zblocks, 0, _s[0]>>>(nslices, zflux, d_extra_Z[0], d_extra_Z[1], d_extra_Z[2], d_extra_Z[3], d_extra_Z[4]);
        GPU::profiler.pop_stopCUDA();

        /* _xflux<<<xgrid, xblocks, 0, _s[0]>>>(nslices, 0, xghostL, xghostR, xflux, d_extra_X[0], d_extra_X[1], d_extra_X[2], d_extra_X[3], d_extra_X[4]); */
        /* _yflux<<<ygrid, yblocks, 0, _s[1]>>>(nslices, 0, yghostL, yghostR, yflux, d_extra_Y[0], d_extra_Y[1], d_extra_Y[2], d_extra_Y[3], d_extra_Y[4]); */
        /* _zflux<<<zgrid, zblocks, 0, _s[2]>>>(nslices, zflux, d_extra_Z[0], d_extra_Z[1], d_extra_Z[2], d_extra_Z[3], d_extra_Z[4]); */

        cudaDeviceSynchronize();

        for (int i = 0; i < 3; ++i)
            cudaStreamDestroy(_s[i]);
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


void GPU::bind_textures()
{
#ifndef _MUTE_GPU_
    _bindTexture(&texR, d_GPUin[0]);
    _bindTexture(&texU, d_GPUin[1]);
    _bindTexture(&texV, d_GPUin[2]);
    _bindTexture(&texW, d_GPUin[3]);
    _bindTexture(&texE, d_GPUin[4]);
    _bindTexture(&texG, d_GPUin[5]);
    _bindTexture(&texP, d_GPUin[6]);
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

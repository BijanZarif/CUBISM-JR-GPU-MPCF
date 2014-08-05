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
        const Real * const Gm, const Real * const Gp,
        const Real * const Pm, const Real * const Pp,
        const Real * const vel,
        Real * const sumG, Real * const sumP, Real * const divU)
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
        const Real * const Gm, const Real * const Gp,
        const Real * const Pm, const Real * const Pp,
        const Real * const vel,
        Real * const sumG, Real * const sumP, Real * const divU)
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
        const Real * const Gm, const Real * const Gp,
        const Real * const Pm, const Real * const Pp,
        const Real * const vel,
        Real * const sumG, Real * const sumP, Real * const divU)
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


__device__
inline void _load_internal_X(const uint_t ix, const uint_t iy, const uint_t iz,
        Real * const r,
        Real * const u,
        Real * const v,
        Real * const w,
        Real * const e,
        Real * const G,
        Real * const P,
        const uint_t dummy1,
        const devPtrSet * const dummy2)
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
}



template <uint_t ix0, uint_t haloStart, uint_t texStart, uint_t ghostStart>
__device__ inline void _load_1X(const uint_t dummy, const uint_t iy, const uint_t iz,
        Real * const r,
        Real * const u,
        Real * const v,
        Real * const w,
        Real * const e,
        Real * const G,
        Real * const P,
        const uint_t global_iz,
        const devPtrSet * const ghost)
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
}

template <uint_t ix0, uint_t haloStart, uint_t texStart, uint_t ghostStart>
__device__ inline void _load_2X(const uint_t dummy, const uint_t iy, const uint_t iz,
        Real * const r,
        Real * const u,
        Real * const v,
        Real * const w,
        Real * const e,
        Real * const G,
        Real * const P,
        const uint_t global_iz,
        const devPtrSet * const ghost)
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
}


template <uint_t ix0, uint_t haloStart, uint_t texStart, uint_t ghostStart>
__device__ inline void _load_3X(const uint_t dummy, const uint_t iy, const uint_t iz,
        Real * const r,
        Real * const u,
        Real * const v,
        Real * const w,
        Real * const e,
        Real * const G,
        Real * const P,
        const uint_t global_iz,
        const devPtrSet * const ghost)
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
}



template <uint_t ix, uint_t haloStart, uint_t texStart, uint_t texEnd>
__device__ inline void _load_halo_stencil_X(
        const uint_t dummy,
        const uint_t iy,
        const uint_t iz,
        Real * const r,
        Real * const u,
        Real * const v,
        Real * const w,
        Real * const e,
        Real * const G,
        Real * const P,
        const uint_t global_iz,
        const devPtrSet * const ghost)
{
    assert((ix == 0 && haloStart == 0 && texStart == 3 && texEnd == _STENCIL_WIDTH_) ||
           (ix == NX-_STENCIL_WIDTH_+3 && haloStart == _STENCIL_WIDTH_-3 && texStart == 0 && texEnd == _STENCIL_WIDTH_-3));

    // Read GMEM first, since latency is higher
#pragma unroll 3
    for (uint_t halo = haloStart; halo < haloStart+3; ++halo)
    {
        const uint_t gid = GHOSTMAPX(halo - haloStart, iy, iz-3+global_iz);
        r[halo] = ghost->r[gid];
        u[halo] = ghost->u[gid];
        v[halo] = ghost->v[gid];
        w[halo] = ghost->w[gid];
        e[halo] = ghost->e[gid];
        G[halo] = ghost->G[gid];
        P[halo] = ghost->P[gid];
        assert(r[halo] >  0.0f);
        assert(e[halo] >  0.0f);
        assert(G[halo] >  0.0f);
        assert(P[halo] >= 0.0f);
    }

    // Now textures
#pragma unroll 5
    for (uint_t tex = texStart; tex < texEnd; ++tex)
    {
        const uint_t sid = ix + tex - texStart;
        r[tex] = tex3D(texR, sid, iy, iz);
        u[tex] = tex3D(texU, sid, iy, iz);
        v[tex] = tex3D(texV, sid, iy, iz);
        w[tex] = tex3D(texW, sid, iy, iz);
        e[tex] = tex3D(texE, sid, iy, iz);
        G[tex] = tex3D(texG, sid, iy, iz);
        P[tex] = tex3D(texP, sid, iy, iz);
        assert(r[tex] >  0.0f);
        assert(e[tex] >  0.0f);
        assert(G[tex] >  0.0f);
        assert(P[tex] >= 0.0f);
    }
}


template <uint_t OFFSET>
__device__ inline void _load_internal_stencil_X(
        const uint_t ix,
        const uint_t iy,
        const uint_t iz,
        Real * const r,
        Real * const u,
        Real * const v,
        Real * const w,
        Real * const e,
        Real * const G,
        Real * const P,
        const uint_t dummy1,
        const devPtrSet * const dummy2)
{
    assert(OFFSET == 3 || OFFSET == _STENCIL_WIDTH_-3);

#pragma unroll 8
    for (uint_t i = 0; i < _STENCIL_WIDTH_; ++i)
    {
        const uint_t sid = ix + i - OFFSET;
        r[i] = tex3D(texR, sid, iy, iz);
        u[i] = tex3D(texU, sid, iy, iz);
        v[i] = tex3D(texV, sid, iy, iz);
        w[i] = tex3D(texW, sid, iy, iz);
        e[i] = tex3D(texE, sid, iy, iz);
        G[i] = tex3D(texG, sid, iy, iz);
        P[i] = tex3D(texP, sid, iy, iz);
        assert(r[i] >  0.0f);
        assert(e[i] >  0.0f);
        assert(G[i] >  0.0f);
        assert(P[i] >= 0.0f);
    }
}


__device__
void _print_stencil(const uint_t bx, const uint_t by, const uint_t tx, const uint_t ty, const Real * const s)
{
    if (bx == blockIdx.x && by == blockIdx.y && tx == threadIdx.x && ty == threadIdx.y)
    {
        printf("Block [%d,%d,%d], Thread [%d,%d,%d], Stencil:\n(",bx,by,blockIdx.z,tx,ty,threadIdx.z);
        for (uint_t i = 0; i < _STENCIL_WIDTH_-1; ++i)
            printf("%f, ", s[i]);
        printf("%f)\n", s[_STENCIL_WIDTH_-1]);
    }
}


__device__
inline void _reconstruct(
        Real * const r,
        Real * const u,
        Real * const v,
        Real * const w,
        Real * const e,
        Real * const G,
        Real * const P)
{
    // Reconstruct 2*_NFLUXES_ values and store the reconstructed value pairs
    // back into the input arrays

    // convert to primitive variables
    for (uint_t i = 0; i < _STENCIL_WIDTH_; ++i)
    {
        e[i] = (e[i] - 0.5f*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i])/r[i] - P[i]) / G[i];
        u[i] = u[i]/r[i];
        v[i] = v[i]/r[i];
        w[i] = w[i]/r[i];
    }

    const Real rm0 = _weno_minus_clipped(r[0], r[1], r[2], r[3], r[4]);
    const Real rp0 = _weno_pluss_clipped(r[1], r[2], r[3], r[4], r[5]);
    const Real rm1 = _weno_minus_clipped(r[1], r[2], r[3], r[4], r[5]);
    const Real rp1 = _weno_pluss_clipped(r[2], r[3], r[4], r[5], r[6]);
    const Real rm2 = _weno_minus_clipped(r[2], r[3], r[4], r[5], r[6]);
    const Real rp2 = _weno_pluss_clipped(r[3], r[4], r[5], r[6], r[7]);

    const Real Gm0 = _weno_minus_clipped(G[0], G[1], G[2], G[3], G[4]);
    const Real Gp0 = _weno_pluss_clipped(G[1], G[2], G[3], G[4], G[5]);
    const Real Gm1 = _weno_minus_clipped(G[1], G[2], G[3], G[4], G[5]);
    const Real Gp1 = _weno_pluss_clipped(G[2], G[3], G[4], G[5], G[6]);
    const Real Gm2 = _weno_minus_clipped(G[2], G[3], G[4], G[5], G[6]);
    const Real Gp2 = _weno_pluss_clipped(G[3], G[4], G[5], G[6], G[7]);

    const Real Pm0 = _weno_minus_clipped(P[0], P[1], P[2], P[3], P[4]);
    const Real Pp0 = _weno_pluss_clipped(P[1], P[2], P[3], P[4], P[5]);
    const Real Pm1 = _weno_minus_clipped(P[1], P[2], P[3], P[4], P[5]);
    const Real Pp1 = _weno_pluss_clipped(P[2], P[3], P[4], P[5], P[6]);
    const Real Pm2 = _weno_minus_clipped(P[2], P[3], P[4], P[5], P[6]);
    const Real Pp2 = _weno_pluss_clipped(P[3], P[4], P[5], P[6], P[7]);

    const Real em0 = _weno_minus_clipped(e[0], e[1], e[2], e[3], e[4]);
    const Real ep0 = _weno_pluss_clipped(e[1], e[2], e[3], e[4], e[5]);
    const Real em1 = _weno_minus_clipped(e[1], e[2], e[3], e[4], e[5]);
    const Real ep1 = _weno_pluss_clipped(e[2], e[3], e[4], e[5], e[6]);
    const Real em2 = _weno_minus_clipped(e[2], e[3], e[4], e[5], e[6]);
    const Real ep2 = _weno_pluss_clipped(e[3], e[4], e[5], e[6], e[7]);

    const Real um0 = _weno_minus_clipped(u[0], u[1], u[2], u[3], u[4]);
    const Real up0 = _weno_pluss_clipped(u[1], u[2], u[3], u[4], u[5]);
    const Real um1 = _weno_minus_clipped(u[1], u[2], u[3], u[4], u[5]);
    const Real up1 = _weno_pluss_clipped(u[2], u[3], u[4], u[5], u[6]);
    const Real um2 = _weno_minus_clipped(u[2], u[3], u[4], u[5], u[6]);
    const Real up2 = _weno_pluss_clipped(u[3], u[4], u[5], u[6], u[7]);

    const Real vm0 = _weno_minus_clipped(v[0], v[1], v[2], v[3], v[4]);
    const Real vp0 = _weno_pluss_clipped(v[1], v[2], v[3], v[4], v[5]);
    const Real vm1 = _weno_minus_clipped(v[1], v[2], v[3], v[4], v[5]);
    const Real vp1 = _weno_pluss_clipped(v[2], v[3], v[4], v[5], v[6]);
    const Real vm2 = _weno_minus_clipped(v[2], v[3], v[4], v[5], v[6]);
    const Real vp2 = _weno_pluss_clipped(v[3], v[4], v[5], v[6], v[7]);

    const Real wm0 = _weno_minus_clipped(w[0], w[1], w[2], w[3], w[4]);
    const Real wp0 = _weno_pluss_clipped(w[1], w[2], w[3], w[4], w[5]);
    const Real wm1 = _weno_minus_clipped(w[1], w[2], w[3], w[4], w[5]);
    const Real wp1 = _weno_pluss_clipped(w[2], w[3], w[4], w[5], w[6]);
    const Real wm2 = _weno_minus_clipped(w[2], w[3], w[4], w[5], w[6]);
    const Real wp2 = _weno_pluss_clipped(w[3], w[4], w[5], w[6], w[7]);

    r[0] = rm0; r[1] = rp0;
    r[2] = rm1; r[3] = rp1;
    r[4] = rm2; r[5] = rp2;

    G[0] = Gm0; G[1] = Gp0;
    G[2] = Gm1; G[3] = Gp1;
    G[4] = Gm2; G[5] = Gp2;

    P[0] = Pm0; P[1] = Pp0;
    P[2] = Pm1; P[3] = Pp1;
    P[4] = Pm2; P[5] = Pp2;

    e[0] = em0; e[1] = ep0;
    e[2] = em1; e[3] = ep1;
    e[4] = em2; e[5] = ep2;

    u[0] = um0; u[1] = up0;
    u[2] = um1; u[3] = up1;
    u[4] = um2; u[5] = up2;

    v[0] = vm0; v[1] = vp0;
    v[2] = vm1; v[3] = vp1;
    v[4] = vm2; v[5] = vp2;

    w[0] = wm0; w[1] = wp0;
    w[2] = wm1; w[3] = wp1;
    w[4] = wm2; w[5] = wp2;
}


typedef void (*ReadFunction)(const uint_t, const uint_t, const uint_t,
        Real * const, Real * const, Real * const, Real * const, Real * const, Real * const, Real * const,
        const uint_t, const devPtrSet * const);




#define _STENCIL_WIDTH_UNIT_ 6
#define _TILE_ 8


__device__
inline void _get_stencil_z(const uint_t ix, const uint_t iy, const uint_t iz,
        Real * __restrict__ stencil, const texture<float, 3, cudaReadModeElementType> tex)
{
    stencil[0] = tex3D(tex, ix, iy, iz-3);
    stencil[1] = tex3D(tex, ix, iy, iz-2);
    stencil[2] = tex3D(tex, ix, iy, iz-1);
    stencil[3] = tex3D(tex, ix, iy, iz);
    stencil[4] = tex3D(tex, ix, iy, iz+1);
    stencil[5] = tex3D(tex, ix, iy, iz+2);
    assert(!isnan(stencil[0]));
    assert(!isnan(stencil[1]));
    assert(!isnan(stencil[2]));
    assert(!isnan(stencil[3]));
    assert(!isnan(stencil[4]));
    assert(!isnan(stencil[5]));
}

__device__
inline void _compute_zflux(const uint_t ix, const uint_t iy, const uint_t iz,
        Real * __restrict__ flux_out, Real * __restrict__ advection_out)
{
    Real r[6], e[6], u[6], v[6], w[6], G[6], P[6];
    _get_stencil_z(ix, iy, iz, r, texR);
    _get_stencil_z(ix, iy, iz, e, texE);
    _get_stencil_z(ix, iy, iz, G, texG);
    _get_stencil_z(ix, iy, iz, P, texP);
    _get_stencil_z(ix, iy, iz, u, texU);
    _get_stencil_z(ix, iy, iz, v, texV);
    _get_stencil_z(ix, iy, iz, w, texW);

    // reconstruct
#pragma unroll 6
    for (uint_t i = 0; i < 6; ++i) // convert to primitive variables
    {
        e[i] = (e[i] - 0.5f*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i])/r[i] - P[i]) / G[i]; //pressure
        u[i] = u[i]/r[i]; // U
        v[i] = v[i]/r[i]; // V
        w[i] = w[i]/r[i]; // W
    }
    const Real pm = _weno_minus_clipped(e[0], e[1], e[2], e[3], e[4]);
    const Real pp = _weno_pluss_clipped(e[1], e[2], e[3], e[4], e[5]);

    const Real um = _weno_minus_clipped(u[0], u[1], u[2], u[3], u[4]);
    const Real up = _weno_pluss_clipped(u[1], u[2], u[3], u[4], u[5]);

    const Real vm = _weno_minus_clipped(v[0], v[1], v[2], v[3], v[4]);
    const Real vp = _weno_pluss_clipped(v[1], v[2], v[3], v[4], v[5]);

    const Real wm = _weno_minus_clipped(w[0], w[1], w[2], w[3], w[4]);
    const Real wp = _weno_pluss_clipped(w[1], w[2], w[3], w[4], w[5]);

    const Real Gm = _weno_minus_clipped(G[0], G[1], G[2], G[3], G[4]);
    const Real Gp = _weno_pluss_clipped(G[1], G[2], G[3], G[4], G[5]);

    const Real Pm = _weno_minus_clipped(P[0], P[1], P[2], P[3], P[4]);
    const Real Pp = _weno_pluss_clipped(P[1], P[2], P[3], P[4], P[5]);

    const Real rm = _weno_minus_clipped(r[0], r[1], r[2], r[3], r[4]);
    const Real rp = _weno_pluss_clipped(r[1], r[2], r[3], r[4], r[5]);

    // assign advection output (partial)
    advection_out[0] = Gm;
    advection_out[1] = Gp;
    advection_out[2] = Pm;
    advection_out[3] = Pp;

    // characteristic velocity
    Real sm, sp;
    _char_vel_einfeldt(rm, rp, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp);
    const Real ss = _char_vel_star(rm, rp, wm, wp, pm, pp, sm, sp);
    assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

    advection_out[4] = _extraterm_hllc_vel(wm, wp, Gm, Gp, Pm, Pp, sm, sp, ss);

    // compute fluxes
    const Real fr = _hllc_rho(rm, rp, wm, wp, sm, sp, ss);
    const Real fu = _hllc_vel(rm, rp, um, up, wm, wp, sm, sp, ss);
    const Real fv = _hllc_vel(rm, rp, vm, vp, wm, wp, sm, sp, ss);
    const Real fw = _hllc_pvel(rm, rp, wm, wp, pm, pp, sm, sp, ss);
    const Real fe = _hllc_e(rm, rp, wm, wp, um, up, vm, vp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss);
    const Real fG = _hllc_rho(Gm, Gp, wm, wp, sm, sp, ss);
    const Real fP = _hllc_rho(Pm, Pp, wm, wp, sm, sp, ss);
    flux_out[0] = fr;
    flux_out[1] = fu;
    flux_out[2] = fv;
    flux_out[3] = fw;
    flux_out[4] = fe;
    flux_out[5] = fG;
    flux_out[6] = fP;
}


__global__
void _divF(const uint_t nslices, const uint_t global_iz,
        const devPtrSet& xghostL, const devPtrSet& xghostR,
        const devPtrSet& yghostL, const devPtrSet& yghostR,
        devPtrSet& rhs)
{
    const uint_t ix = blockIdx.x * _TILE_ + threadIdx.x;
    const uint_t iy = blockIdx.y * _TILE_ + threadIdx.y;

    const uint_t tid = blockDim.x*threadIdx.y + threadIdx.x;

    __shared__ Real s_stencil[_TILE_][_TILE_+6];
    __shared__ Real s_flux[_TILE_][_TILE_+1];
    __shared__ Real s_Gm[_TILE_][_TILE_+1];
    __shared__ Real s_Gp[_TILE_][_TILE_+1];
    __shared__ Real s_Pm[_TILE_][_TILE_+1];
    __shared__ Real s_Pp[_TILE_][_TILE_+1];
    __shared__ Real s_hllc_vel[_TILE_][_TILE_+1];

    // assumes NX % _TILE_ = 0 and NY % _TILE_ = 0

    Real _rhs=0.0f, _sumG=0.0f, _sumP=0.0f, _divU=0.0f;

    /* *
     * 1. Compute z-fluxes for first slice before enter loop
     * */
    Real _zflux_m[7]; // z-flux minus, 7 flow quantities
    Real _zadvection_m[5]; // Gm, Gp, Pm, Pp, hllc_vel
    _compute_zflux(ix, iy, 3, _zflux_m, _zadvection_m);

    /* *
     * 2. Loop over x-y-slices
     * */
    for (uint_t iz = 3; iz < nslices+3; ++iz) // first and last 3 slices are zghosts
    {
        // get s_stencil into shared mem

        // __syncthreads()
        // Real myStencil
    }

}


__global__
void _xflux_mod(const uint_t nslices, const uint_t global_iz,
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

    if (ix < NXP1 && iy < NY)
    {
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

            // stencils (7 * _STENCIL_WIDTH_ registers per thread)
            Real r[_STENCIL_WIDTH_UNIT_];
            Real u[_STENCIL_WIDTH_UNIT_];
            Real v[_STENCIL_WIDTH_UNIT_];
            Real w[_STENCIL_WIDTH_UNIT_];
            Real e[_STENCIL_WIDTH_UNIT_];
            Real G[_STENCIL_WIDTH_UNIT_];
            Real P[_STENCIL_WIDTH_UNIT_];

/* #pragma unroll 6 */
/*             for (int i = 0; i < _STENCIL_WIDTH_UNIT_; ++i) */
/*             { */
/*                 r[i] = 1.0f; */
/*                 u[i] = 0.0f; */
/*                 v[i] = 0.0f; */
/*                 w[i] = 0.0f; */
/*                 e[i] = 1.0f; */
/*                 G[i] = 2.5f; */
/*                 P[i] = 0.0f; */
/*             } */
            // 1.)
            if (0 == ix)
                _load_3X<0,0,3,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL);
            else if (1 == ix)
                _load_2X<0,0,2,1>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL);
            else if (2 == ix)
                _load_1X<0,0,1,2>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostL);
            else if (NXP1-3 == ix)
                _load_1X<NXP1-6,5,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NXP1-2 == ix)
                _load_2X<NXP1-5,4,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else if (NXP1-1 == ix)
                _load_3X<NXP1-4,3,0,0>(ix, iy, iz, r, u, v, w, e, G, P, global_iz, &ghostR);
            else
                _load_internal_X(ix, iy, iz, r, u, v, w, e, G, P, global_iz, NULL);

            // 2.)
            // convert to primitive variables
#pragma unroll 6
            for (uint_t i = 0; i < _STENCIL_WIDTH_UNIT_; ++i)
            {
                e[i] = (e[i] - 0.5f*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i])/r[i] - P[i]) / G[i];
                u[i] = u[i]/r[i];
                v[i] = v[i]/r[i];
                w[i] = w[i]/r[i];
            }

            const Real rm = _weno_minus_clipped(r[0], r[1], r[2], r[3], r[4]);
            const Real rp = _weno_pluss_clipped(r[1], r[2], r[3], r[4], r[5]);
            assert(!isnan(rp)); assert(!isnan(rm));

            const Real Gm = _weno_minus_clipped(G[0], G[1], G[2], G[3], G[4]);
            const Real Gp = _weno_pluss_clipped(G[1], G[2], G[3], G[4], G[5]);
            assert(!isnan(Gp)); assert(!isnan(Gm));

            const Real Pm = _weno_minus_clipped(P[0], P[1], P[2], P[3], P[4]);
            const Real Pp = _weno_pluss_clipped(P[1], P[2], P[3], P[4], P[5]);
            assert(!isnan(Pp)); assert(!isnan(Pm));

            const Real pm = _weno_minus_clipped(e[0], e[1], e[2], e[3], e[4]);
            const Real pp = _weno_pluss_clipped(e[1], e[2], e[3], e[4], e[5]);
            assert(!isnan(pp)); assert(!isnan(pm));

            const Real um = _weno_minus_clipped(u[0], u[1], u[2], u[3], u[4]);
            const Real up = _weno_pluss_clipped(u[1], u[2], u[3], u[4], u[5]);
            assert(!isnan(up)); assert(!isnan(um));

            const Real vm = _weno_minus_clipped(v[0], v[1], v[2], v[3], v[4]);
            const Real vp = _weno_pluss_clipped(v[1], v[2], v[3], v[4], v[5]);
            assert(!isnan(vp)); assert(!isnan(vm));

            const Real wm = _weno_minus_clipped(w[0], w[1], w[2], w[3], w[4]);
            const Real wp = _weno_pluss_clipped(w[1], w[2], w[3], w[4], w[5]);
            assert(!isnan(wp)); assert(!isnan(wm));

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

            const Real hllc_vel = _extraterm_hllc_vel(um, up, Gm, Gp, Pm, Pp, sm, sp, ss);

            /* const uint_t idx = ID3(iy, ix, iz-3, NY, NXP1); */
            /* flux.r[idx] = fr + fu + fv + fw + fe + fG + fP + hllc_vel; */

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
            /* r.im3 = r.im2 = r.im1 = r.i = r.ip1 = r.ip2 = 1.0f; */
            /* u.im3 = u.im2 = u.im1 = u.i = u.ip1 = u.ip2 = 0.0f; */
            /* v.im3 = v.im2 = v.im1 = v.i = v.ip1 = v.ip2 = 0.0f; */
            /* w.im3 = w.im2 = w.im1 = w.i = w.ip1 = w.ip2 = 0.0f; */
            /* e.im3 = e.im2 = e.im1 = e.i = e.ip1 = e.ip2 = 1.0f; */
            /* G.im3 = G.im2 = G.im1 = G.i = G.ip1 = G.ip2 = 2.5f; */
            /* P.im3 = P.im2 = P.im1 = P.i = P.ip1 = P.ip2 = 0.0f; */
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

            const Real hllc_vel = _extraterm_hllc_vel(um, up, Gm, Gp, Pm, Pp, sm, sp, ss);

            /* const uint_t idx = ID3(iy, ix, iz-3, NY, NXP1); */
            /* flux.r[idx] = fr + fu + fv + fw + fe + fG + fP + hllc_vel; */

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
void _yflux(const uint_t nslices, const uint_t global_iz,
        devPtrSet ghostL, devPtrSet ghostR, devPtrSet flux,
        Real * const xtra_vel,
        Real * const xtra_Gm, Real * const xtra_Gp,
        Real * const xtra_Pm, Real * const xtra_Pp)
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

    /* *
     * The following code requires 1D thread blocks with dimension
     * dim3(_NTHREADS_, 1, 1)
     * on a 2D grid with dimension
     * dim3((NX + _NTHREADS_ - 1) / _NTHREADS_, NYP1, 1)
     *
     * The load of ghosts is organized into a switch block with 6 cases.
     *
     * NOTE: To minimize the switch cases to 6 (and simplify code) the
     * following requires that NY >= 5
     * */
    assert(NYP1 > 5);

#if 1
    if (ix < NX && iy < NYP1)
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
            _read_stencil_Y(r, texR, ghostL.r, ghostR.r, ix, iy, iz, global_iz);
            _read_stencil_Y(u, texU, ghostL.u, ghostR.u, ix, iy, iz, global_iz);
            _read_stencil_Y(v, texV, ghostL.v, ghostR.v, ix, iy, iz, global_iz);
            _read_stencil_Y(w, texW, ghostL.w, ghostR.w, ix, iy, iz, global_iz);
            _read_stencil_Y(e, texE, ghostL.e, ghostR.e, ix, iy, iz, global_iz);
            _read_stencil_Y(G, texG, ghostL.G, ghostR.G, ix, iy, iz, global_iz);
            _read_stencil_Y(P, texP, ghostL.P, ghostR.P, ix, iy, iz, global_iz);
#else
            // less conditionals, additional tex3D fetches
            _load_stencil_tex3D_Y(r, texR, ix, iy, iz);
            _load_stencil_tex3D_Y(u, texU, ix, iy, iz);
            _load_stencil_tex3D_Y(v, texV, ix, iy, iz);
            _load_stencil_tex3D_Y(w, texW, ix, iy, iz);
            _load_stencil_tex3D_Y(e, texE, ix, iy, iz);
            _load_stencil_tex3D_Y(G, texG, ix, iy, iz);
            _load_stencil_tex3D_Y(P, texP, ix, iy, iz);
            switch (iy)
            {
                case 0:
                    _load_3ghosts_Y(r.im3, r.im2, r.im1, ghostL.r, ix, iz-3+global_iz);
                    _load_3ghosts_Y(u.im3, u.im2, u.im1, ghostL.u, ix, iz-3+global_iz);
                    _load_3ghosts_Y(v.im3, v.im2, v.im1, ghostL.v, ix, iz-3+global_iz);
                    _load_3ghosts_Y(w.im3, w.im2, w.im1, ghostL.w, ix, iz-3+global_iz);
                    _load_3ghosts_Y(e.im3, e.im2, e.im1, ghostL.e, ix, iz-3+global_iz);
                    _load_3ghosts_Y(G.im3, G.im2, G.im1, ghostL.G, ix, iz-3+global_iz);
                    _load_3ghosts_Y(P.im3, P.im2, P.im1, ghostL.P, ix, iz-3+global_iz);
                    break;
                case 1:
                    _load_2ghosts_Y(r.im3, r.im2, 1, 2, ghostL.r, ix, iz-3+global_iz);
                    _load_2ghosts_Y(u.im3, u.im2, 1, 2, ghostL.u, ix, iz-3+global_iz);
                    _load_2ghosts_Y(v.im3, v.im2, 1, 2, ghostL.v, ix, iz-3+global_iz);
                    _load_2ghosts_Y(w.im3, w.im2, 1, 2, ghostL.w, ix, iz-3+global_iz);
                    _load_2ghosts_Y(e.im3, e.im2, 1, 2, ghostL.e, ix, iz-3+global_iz);
                    _load_2ghosts_Y(G.im3, G.im2, 1, 2, ghostL.G, ix, iz-3+global_iz);
                    _load_2ghosts_Y(P.im3, P.im2, 1, 2, ghostL.P, ix, iz-3+global_iz);
                    break;
                case 2:
                    _load_1ghost_Y(r.im3, 2, ghostL.r, ix, iz-3+global_iz);
                    _load_1ghost_Y(u.im3, 2, ghostL.u, ix, iz-3+global_iz);
                    _load_1ghost_Y(v.im3, 2, ghostL.v, ix, iz-3+global_iz);
                    _load_1ghost_Y(w.im3, 2, ghostL.w, ix, iz-3+global_iz);
                    _load_1ghost_Y(e.im3, 2, ghostL.e, ix, iz-3+global_iz);
                    _load_1ghost_Y(G.im3, 2, ghostL.G, ix, iz-3+global_iz);
                    _load_1ghost_Y(P.im3, 2, ghostL.P, ix, iz-3+global_iz);
                    break;
                case (NYP1-3):
                    _load_1ghost_Y(r.ip2, 0, ghostR.r, ix, iz-3+global_iz);
                    _load_1ghost_Y(u.ip2, 0, ghostR.u, ix, iz-3+global_iz);
                    _load_1ghost_Y(v.ip2, 0, ghostR.v, ix, iz-3+global_iz);
                    _load_1ghost_Y(w.ip2, 0, ghostR.w, ix, iz-3+global_iz);
                    _load_1ghost_Y(e.ip2, 0, ghostR.e, ix, iz-3+global_iz);
                    _load_1ghost_Y(G.ip2, 0, ghostR.G, ix, iz-3+global_iz);
                    _load_1ghost_Y(P.ip2, 0, ghostR.P, ix, iz-3+global_iz);
                    break;
                case (NYP1-2):
                    _load_2ghosts_Y(r.ip1, r.ip2, 0, 1, ghostR.r, ix, iz-3+global_iz);
                    _load_2ghosts_Y(u.ip1, u.ip2, 0, 1, ghostR.u, ix, iz-3+global_iz);
                    _load_2ghosts_Y(v.ip1, v.ip2, 0, 1, ghostR.v, ix, iz-3+global_iz);
                    _load_2ghosts_Y(w.ip1, w.ip2, 0, 1, ghostR.w, ix, iz-3+global_iz);
                    _load_2ghosts_Y(e.ip1, e.ip2, 0, 1, ghostR.e, ix, iz-3+global_iz);
                    _load_2ghosts_Y(G.ip1, G.ip2, 0, 1, ghostR.G, ix, iz-3+global_iz);
                    _load_2ghosts_Y(P.ip1, P.ip2, 0, 1, ghostR.P, ix, iz-3+global_iz);
                    break;
                case (NYP1-1):
                    _load_3ghosts_Y(r.i, r.ip1, r.ip2, ghostR.r, ix, iz-3+global_iz);
                    _load_3ghosts_Y(u.i, u.ip1, u.ip2, ghostR.u, ix, iz-3+global_iz);
                    _load_3ghosts_Y(v.i, v.ip1, v.ip2, ghostR.v, ix, iz-3+global_iz);
                    _load_3ghosts_Y(w.i, w.ip1, w.ip2, ghostR.w, ix, iz-3+global_iz);
                    _load_3ghosts_Y(e.i, e.ip1, e.ip2, ghostR.e, ix, iz-3+global_iz);
                    _load_3ghosts_Y(G.i, G.ip1, G.ip2, ghostR.G, ix, iz-3+global_iz);
                    _load_3ghosts_Y(P.i, P.ip1, P.ip2, ghostR.P, ix, iz-3+global_iz);
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
            _char_vel_einfeldt(rm, rp, vm, vp, pm, pp, Gm, Gp, Pm, Pp, sm, sp);
            const Real ss = _char_vel_star(rm, rp, vm, vp, pm, pp, sm, sp);
            assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

            // 4.)
            const Real fr = _hllc_rho(rm, rp, vm, vp, sm, sp, ss);
            const Real fu = _hllc_vel(rm, rp, um, up, vm, vp, sm, sp, ss);
            const Real fv = _hllc_pvel(rm, rp, vm, vp, pm, pp, sm, sp, ss);
            const Real fw = _hllc_vel(rm, rp, wm, wp, vm, vp, sm, sp, ss);
            const Real fe = _hllc_e(rm, rp, vm, vp, um, up, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss);
            const Real fG = _hllc_rho(Gm, Gp, vm, vp, sm, sp, ss);
            const Real fP = _hllc_rho(Pm, Pp, vm, vp, sm, sp, ss);
            assert(!isnan(fr)); assert(!isnan(fu)); assert(!isnan(fv)); assert(!isnan(fw)); assert(!isnan(fe)); assert(!isnan(fG)); assert(!isnan(fP));

            const uint_t idx = ID3(ix, iy, iz-3, NX, NYP1);
            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            // 5.)
            xtra_vel[idx] = _extraterm_hllc_vel(vm, vp, Gm, Gp, Pm, Pp, sm, sp, ss);
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
#endif

#if 0
    if (ix < NX && iy < NYP1)
    {
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
            _yfetch_data(texR, ghostL.r, ghostR.r, ix, iy, iz, global_iz, NX, NYP1, rm3, rm2, rm1, rp1, rp2, rp3);
            assert(rm3 > 0); assert(rm2 > 0); assert(rm1 > 0); assert(rp1 > 0); assert(rp2 > 0); assert(rp3 > 0);

            Real um3, um2, um1, up1, up2, up3;
            _yfetch_data(texU, ghostL.u, ghostR.u, ix, iy, iz, global_iz, NX, NYP1, um3, um2, um1, up1, up2, up3);

            Real vm3, vm2, vm1, vp1, vp2, vp3;
            _yfetch_data(texV, ghostL.v, ghostR.v, ix, iy, iz, global_iz, NX, NYP1, vm3, vm2, vm1, vp1, vp2, vp3);

            Real wm3, wm2, wm1, wp1, wp2, wp3;
            _yfetch_data(texW, ghostL.w, ghostR.w, ix, iy, iz, global_iz, NX, NYP1, wm3, wm2, wm1, wp1, wp2, wp3);

            Real em3, em2, em1, ep1, ep2, ep3;
            _yfetch_data(texE, ghostL.e, ghostR.e, ix, iy, iz, global_iz, NX, NYP1, em3, em2, em1, ep1, ep2, ep3);
            assert(em3 > 0); assert(em2 > 0); assert(em1 > 0); assert(ep1 > 0); assert(ep2 > 0); assert(ep3 > 0);

            Real Gm3, Gm2, Gm1, Gp1, Gp2, Gp3;
            _yfetch_data(texG, ghostL.G, ghostR.G, ix, iy, iz, global_iz, NX, NYP1, Gm3, Gm2, Gm1, Gp1, Gp2, Gp3);
            assert(Gm3 > 0); assert(Gm2 > 0); assert(Gm1 > 0); assert(Gp1 > 0); assert(Gp2 > 0); assert(Gp3 > 0);

            Real Pm3, Pm2, Pm1, Pp1, Pp2, Pp3;
            _yfetch_data(texP, ghostL.P, ghostR.P, ix, iy, iz, global_iz, NX, NYP1, Pm3, Pm2, Pm1, Pp1, Pp2, Pp3);
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
            const Real fr = _hllc_rho(rm, rp, vm, vp, sm, sp, ss);
            const Real fu = _hllc_vel(rm, rp, um, up, vm, vp, sm, sp, ss);
            const Real fv = _hllc_pvel(rm, rp, vm, vp, pm, pp, sm, sp, ss);
            const Real fw = _hllc_vel(rm, rp, wm, wp, vm, vp, sm, sp, ss);
            const Real fe = _hllc_e(rm, rp, vm, vp, um, up, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss);
            const Real fG = _hllc_rho(Gm, Gp, vm, vp, sm, sp, ss);
            const Real fP = _hllc_rho(Pm, Pp, vm, vp, sm, sp, ss);

            const uint_t idx = ID3(ix, iy, iz-3, NX, NYP1);
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
#endif
}




__global__
void _zflux(const uint_t nslices, devPtrSet flux,
        Real * const xtra_vel,
        Real * const xtra_Gm, Real * const xtra_Gp,
        Real * const xtra_Pm, Real * const xtra_Pp)
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

    /* *
     * The following code requires 1D thread blocks with dimension
     * dim3(_NTHREADS_, 1, 1)
     * on a 2D grid with dimension
     * dim3((NX + _NTHREADS_ - 1) / _NTHREADS_, NY, 1)
     *
     * The following requires that NZ > 0
     * */
    assert(NodeBlock::sizeZ > 0);

#if 1
    if (ix < NX && iy < NY)
    {
        Stencil r, u, v, w, e, G, P;
        Stencil p; // for reconstruction
        for (uint_t iz = 3; iz < (nslices+1)+3; ++iz) // first and last 3 slices are zghosts; need to compute nslices+1 fluxes in z-direction
        {
            /* *
             * The general task order is (for each chunk slice along NZ):
             * 1.) Load tex3D into stencil (do this 7x, for each quantity)
             * 2.) Reconstruct primitive values using WENO5/WENO3
             * 3.) Compute characteristic velocities
             * 4.) Compute fluxes
             * 5.) Compute RHS for advection of G and P
             * */

            // 1.)
            // that was easy!
            _read_stencil_Z(r, texR, ix, iy, iz);
            _read_stencil_Z(u, texU, ix, iy, iz);
            _read_stencil_Z(v, texV, ix, iy, iz);
            _read_stencil_Z(w, texW, ix, iy, iz);
            _read_stencil_Z(e, texE, ix, iy, iz);
            _read_stencil_Z(G, texG, ix, iy, iz);
            _read_stencil_Z(P, texP, ix, iy, iz);

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
            _char_vel_einfeldt(rm, rp, wm, wp, pm, pp, Gm, Gp, Pm, Pp, sm, sp);
            const Real ss = _char_vel_star(rm, rp, wm, wp, pm, pp, sm, sp);
            assert(!isnan(sm)); assert(!isnan(sp)); assert(!isnan(ss));

            // 4.)
            const Real fr = _hllc_rho(rm, rp, wm, wp, sm, sp, ss);
            const Real fu = _hllc_vel(rm, rp, um, up, wm, wp, sm, sp, ss);
            const Real fv = _hllc_vel(rm, rp, vm, vp, wm, wp, sm, sp, ss);
            const Real fw = _hllc_pvel(rm, rp, wm, wp, pm, pp, sm, sp, ss);
            const Real fe = _hllc_e(rm, rp, wm, wp, um, up, vm, vp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss);
            const Real fG = _hllc_rho(Gm, Gp, wm, wp, sm, sp, ss);
            const Real fP = _hllc_rho(Pm, Pp, wm, wp, sm, sp, ss);

            const uint_t idx = ID3(ix, iy, iz-3, NX, NY);
            flux.r[idx] = fr;
            flux.u[idx] = fu;
            flux.v[idx] = fv;
            flux.w[idx] = fw;
            flux.e[idx] = fe;
            flux.G[idx] = fG;
            flux.P[idx] = fP;

            // 5.)
            xtra_vel[idx] = _extraterm_hllc_vel(wm, wp, Gm, Gp, Pm, Pp, sm, sp, ss);
            xtra_Gm[idx]  = Gm;
            xtra_Gp[idx]  = Gp;
            xtra_Pm[idx]  = Pm;
            xtra_Pp[idx]  = Pp;
        }
    }
#endif


#if 0
    if (ix < NX && iy < NY)
    {
        // need to compute nslices+1 fluxes in z-direction
        for (uint_t iz = 3; iz < (nslices+1)+3; ++iz)
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
            const Real fr = _hllc_rho(rm, rp, wm, wp, sm, sp, ss);
            const Real fu = _hllc_vel(rm, rp, um, up, wm, wp, sm, sp, ss);
            const Real fv = _hllc_vel(rm, rp, vm, vp, wm, wp, sm, sp, ss);
            const Real fw = _hllc_pvel(rm, rp, wm, wp, pm, pp, sm, sp, ss);
            const Real fe = _hllc_e(rm, rp, wm, wp, um, up, vm, vp, pm, pp, Gm, Gp, Pm, Pp, sm, sp, ss);
            const Real fG = _hllc_rho(Gm, Gp, wm, wp, sm, sp, ss);
            const Real fP = _hllc_rho(Pm, Pp, wm, wp, sm, sp, ss);

            const uint_t idx = ID3(ix, iy, iz-3, NX, NY);
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
#endif
}


__global__
void _divergence(const uint_t nslices,
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
void _update(const uint_t nslices, const Real b, devPtrSet tmp, const devPtrSet rhs)
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
    devPtrSet xghostL(d_xgl);
    devPtrSet xghostR(d_xgr);
    devPtrSet xflux(d_xflux);

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
    devPtrSet yghostL(d_ygl);
    devPtrSet yghostR(d_ygr);
    devPtrSet yflux(d_yflux);

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
    devPtrSet zflux(d_zflux);

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

    devPtrSet xflux(d_xflux);
    devPtrSet yflux(d_yflux);
    devPtrSet zflux(d_zflux);
    devPtrSet rhs(d_rhs);
    devPtrSet tmp(d_tmp);

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
    devPtrSet tmp(d_tmp);
    devPtrSet rhs(d_rhs);

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
    devPtrSet xghostL(d_xgl);
    devPtrSet xghostR(d_xgr);
    devPtrSet xflux(d_xflux);

    devPtrSet yghostL(d_ygl);
    devPtrSet yghostR(d_ygr);
    devPtrSet yflux(d_yflux);

    devPtrSet zflux(d_zflux);

    devPtrSet rhs(d_rhs);

    {
        const uint_t nslices = NodeBlock::sizeZ;

        const dim3 xblocks(1, _NTHREADS_, 1);
        const dim3 yblocks(_NTHREADS_, 1, 1);
        const dim3 zblocks(_NTHREADS_, 1, 1);
        const dim3 xgrid_half((NXP1/2 + _NFLUXES_ - 1)/_NFLUXES_, (NY + _NTHREADS_ - 1) / _NTHREADS_,   1);
        const dim3 xgrid(NXP1, (NY + _NTHREADS_ - 1) / _NTHREADS_,   1);
        const dim3 ygrid((NX   + _NTHREADS_ - 1) / _NTHREADS_, NYP1, 1);
        const dim3 zgrid((NX   + _NTHREADS_ - 1) / _NTHREADS_, NY,   1);

        /* const dim3 Dblocks(_TILE_, _TILE_, 1); */
        /* const dim3 Dgrid((NX + _TILE_ - 1)/_TILE_, (NY + _TILE_ - 1)/_TILE_, 1); */

        /* GPU::profiler.push_startCUDA("_TEST_KERNEL1"); */
        /* _divF<<<Dgrid, Dblocks>>>(nslices, 0, xghostL, xghostR, yghostL, yghostR, rhs); */
        /* GPU::profiler.pop_stopCUDA(); */

        /* GPU::profiler.push_startCUDA("_TEST_KERNEL1"); */
        /* _xflux_left<<<xgrid_half, xblocks>>>(nslices, 0, xghostL, xflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp); */
        /* GPU::profiler.pop_stopCUDA(); */

        GPU::profiler.push_startCUDA("_TEST_KERNEL2");
        _xflux_mod<<<xgrid, xblocks>>>(nslices, 0, xghostL, xghostR, xflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
        GPU::profiler.pop_stopCUDA();

        GPU::profiler.push_startCUDA("_TEST_KERNEL3");
        /* _xflux_left<<<xgrid, xblocks>>>(nslices, 0, xghostL, xflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp); */
        _xflux<<<xgrid, xblocks>>>(nslices, 0, xghostL, xghostR, xflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
        _yflux<<<ygrid, yblocks>>>(nslices, 0, yghostL, yghostR, yflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
        _zflux<<<zgrid, zblocks>>>(nslices, zflux, d_hllc_vel, d_Gm, d_Gp, d_Pm, d_Pp);
        GPU::profiler.pop_stopCUDA();
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

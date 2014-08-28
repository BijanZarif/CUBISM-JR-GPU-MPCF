/* File        : Texture.cu */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Thu 14 Aug 2014 02:32:34 PM CEST */
/* Modified    : Thu 14 Aug 2014 03:08:46 PM CEST */
/* Description : Texture setup and stencil readers */

// set 0
texture<float, 3, cudaReadModeElementType> texR00;
texture<float, 3, cudaReadModeElementType> texU00;
texture<float, 3, cudaReadModeElementType> texV00;
texture<float, 3, cudaReadModeElementType> texW00;
texture<float, 3, cudaReadModeElementType> texE00;
texture<float, 3, cudaReadModeElementType> texG00;
texture<float, 3, cudaReadModeElementType> texP00;

// set 1
texture<float, 3, cudaReadModeElementType> texR01;
texture<float, 3, cudaReadModeElementType> texU01;
texture<float, 3, cudaReadModeElementType> texV01;
texture<float, 3, cudaReadModeElementType> texW01;
texture<float, 3, cudaReadModeElementType> texE01;
texture<float, 3, cudaReadModeElementType> texG01;
texture<float, 3, cudaReadModeElementType> texP01;

// Stencil loaders
///////////////////////////////////////////////////////////////////////////////
// Reading from 00
__device__
inline void _load_internal_X00(const uint_t ix, const uint_t iy, const uint_t iz,
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
    r[0]  = tex3D(texR00, ix-3, iy, iz);
    r[1]  = tex3D(texR00, ix-2, iy, iz);
    r[2]  = tex3D(texR00, ix-1, iy, iz);
    r[3]  = tex3D(texR00, ix,   iy, iz);
    r[4]  = tex3D(texR00, ix+1, iy, iz);
    r[5]  = tex3D(texR00, ix+2, iy, iz);

    u[0]  = tex3D(texU00, ix-3, iy, iz);
    u[1]  = tex3D(texU00, ix-2, iy, iz);
    u[2]  = tex3D(texU00, ix-1, iy, iz);
    u[3]  = tex3D(texU00, ix,   iy, iz);
    u[4]  = tex3D(texU00, ix+1, iy, iz);
    u[5]  = tex3D(texU00, ix+2, iy, iz);

    v[0]  = tex3D(texV00, ix-3, iy, iz);
    v[1]  = tex3D(texV00, ix-2, iy, iz);
    v[2]  = tex3D(texV00, ix-1, iy, iz);
    v[3]  = tex3D(texV00, ix,   iy, iz);
    v[4]  = tex3D(texV00, ix+1, iy, iz);
    v[5]  = tex3D(texV00, ix+2, iy, iz);

    w[0]  = tex3D(texW00, ix-3, iy, iz);
    w[1]  = tex3D(texW00, ix-2, iy, iz);
    w[2]  = tex3D(texW00, ix-1, iy, iz);
    w[3]  = tex3D(texW00, ix,   iy, iz);
    w[4]  = tex3D(texW00, ix+1, iy, iz);
    w[5]  = tex3D(texW00, ix+2, iy, iz);

    e[0]  = tex3D(texE00, ix-3, iy, iz);
    e[1]  = tex3D(texE00, ix-2, iy, iz);
    e[2]  = tex3D(texE00, ix-1, iy, iz);
    e[3]  = tex3D(texE00, ix,   iy, iz);
    e[4]  = tex3D(texE00, ix+1, iy, iz);
    e[5]  = tex3D(texE00, ix+2, iy, iz);

    G[0]  = tex3D(texG00, ix-3, iy, iz);
    G[1]  = tex3D(texG00, ix-2, iy, iz);
    G[2]  = tex3D(texG00, ix-1, iy, iz);
    G[3]  = tex3D(texG00, ix,   iy, iz);
    G[4]  = tex3D(texG00, ix+1, iy, iz);
    G[5]  = tex3D(texG00, ix+2, iy, iz);

    P[0]  = tex3D(texP00, ix-3, iy, iz);
    P[1]  = tex3D(texP00, ix-2, iy, iz);
    P[2]  = tex3D(texP00, ix-1, iy, iz);
    P[3]  = tex3D(texP00, ix,   iy, iz);
    P[4]  = tex3D(texP00, ix+1, iy, iz);
    P[5]  = tex3D(texP00, ix+2, iy, iz);

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
inline void _load_internal_Y00(const uint_t ix, const uint_t iy, const uint_t iz,
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
    r[0]  = tex3D(texR00, ix, iy-3, iz);
    r[1]  = tex3D(texR00, ix, iy-2, iz);
    r[2]  = tex3D(texR00, ix, iy-1, iz);
    r[3]  = tex3D(texR00, ix, iy,   iz);
    r[4]  = tex3D(texR00, ix, iy+1, iz);
    r[5]  = tex3D(texR00, ix, iy+2, iz);

    u[0]  = tex3D(texU00, ix, iy-3, iz);
    u[1]  = tex3D(texU00, ix, iy-2, iz);
    u[2]  = tex3D(texU00, ix, iy-1, iz);
    u[3]  = tex3D(texU00, ix, iy,   iz);
    u[4]  = tex3D(texU00, ix, iy+1, iz);
    u[5]  = tex3D(texU00, ix, iy+2, iz);

    v[0]  = tex3D(texV00, ix, iy-3, iz);
    v[1]  = tex3D(texV00, ix, iy-2, iz);
    v[2]  = tex3D(texV00, ix, iy-1, iz);
    v[3]  = tex3D(texV00, ix, iy,   iz);
    v[4]  = tex3D(texV00, ix, iy+1, iz);
    v[5]  = tex3D(texV00, ix, iy+2, iz);

    w[0]  = tex3D(texW00, ix, iy-3, iz);
    w[1]  = tex3D(texW00, ix, iy-2, iz);
    w[2]  = tex3D(texW00, ix, iy-1, iz);
    w[3]  = tex3D(texW00, ix, iy,   iz);
    w[4]  = tex3D(texW00, ix, iy+1, iz);
    w[5]  = tex3D(texW00, ix, iy+2, iz);

    e[0]  = tex3D(texE00, ix, iy-3, iz);
    e[1]  = tex3D(texE00, ix, iy-2, iz);
    e[2]  = tex3D(texE00, ix, iy-1, iz);
    e[3]  = tex3D(texE00, ix, iy,   iz);
    e[4]  = tex3D(texE00, ix, iy+1, iz);
    e[5]  = tex3D(texE00, ix, iy+2, iz);

    G[0]  = tex3D(texG00, ix, iy-3, iz);
    G[1]  = tex3D(texG00, ix, iy-2, iz);
    G[2]  = tex3D(texG00, ix, iy-1, iz);
    G[3]  = tex3D(texG00, ix, iy,   iz);
    G[4]  = tex3D(texG00, ix, iy+1, iz);
    G[5]  = tex3D(texG00, ix, iy+2, iz);

    P[0]  = tex3D(texP00, ix, iy-3, iz);
    P[1]  = tex3D(texP00, ix, iy-2, iz);
    P[2]  = tex3D(texP00, ix, iy-1, iz);
    P[3]  = tex3D(texP00, ix, iy,   iz);
    P[4]  = tex3D(texP00, ix, iy+1, iz);
    P[5]  = tex3D(texP00, ix, iy+2, iz);

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
inline void _load_internal_Z00(const uint_t ix, const uint_t iy, const uint_t iz,
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
    r[0]  = tex3D(texR00, ix, iy, iz-3);
    r[1]  = tex3D(texR00, ix, iy, iz-2);
    r[2]  = tex3D(texR00, ix, iy, iz-1);
    r[3]  = tex3D(texR00, ix, iy, iz);
    r[4]  = tex3D(texR00, ix, iy, iz+1);
    r[5]  = tex3D(texR00, ix, iy, iz+2);

    u[0]  = tex3D(texU00, ix, iy, iz-3);
    u[1]  = tex3D(texU00, ix, iy, iz-2);
    u[2]  = tex3D(texU00, ix, iy, iz-1);
    u[3]  = tex3D(texU00, ix, iy, iz);
    u[4]  = tex3D(texU00, ix, iy, iz+1);
    u[5]  = tex3D(texU00, ix, iy, iz+2);

    v[0]  = tex3D(texV00, ix, iy, iz-3);
    v[1]  = tex3D(texV00, ix, iy, iz-2);
    v[2]  = tex3D(texV00, ix, iy, iz-1);
    v[3]  = tex3D(texV00, ix, iy, iz);
    v[4]  = tex3D(texV00, ix, iy, iz+1);
    v[5]  = tex3D(texV00, ix, iy, iz+2);

    w[0]  = tex3D(texW00, ix, iy, iz-3);
    w[1]  = tex3D(texW00, ix, iy, iz-2);
    w[2]  = tex3D(texW00, ix, iy, iz-1);
    w[3]  = tex3D(texW00, ix, iy, iz);
    w[4]  = tex3D(texW00, ix, iy, iz+1);
    w[5]  = tex3D(texW00, ix, iy, iz+2);

    e[0]  = tex3D(texE00, ix, iy, iz-3);
    e[1]  = tex3D(texE00, ix, iy, iz-2);
    e[2]  = tex3D(texE00, ix, iy, iz-1);
    e[3]  = tex3D(texE00, ix, iy, iz);
    e[4]  = tex3D(texE00, ix, iy, iz+1);
    e[5]  = tex3D(texE00, ix, iy, iz+2);

    G[0]  = tex3D(texG00, ix, iy, iz-3);
    G[1]  = tex3D(texG00, ix, iy, iz-2);
    G[2]  = tex3D(texG00, ix, iy, iz-1);
    G[3]  = tex3D(texG00, ix, iy, iz);
    G[4]  = tex3D(texG00, ix, iy, iz+1);
    G[5]  = tex3D(texG00, ix, iy, iz+2);

    P[0]  = tex3D(texP00, ix, iy, iz-3);
    P[1]  = tex3D(texP00, ix, iy, iz-2);
    P[2]  = tex3D(texP00, ix, iy, iz-1);
    P[3]  = tex3D(texP00, ix, iy, iz);
    P[4]  = tex3D(texP00, ix, iy, iz+1);
    P[5]  = tex3D(texP00, ix, iy, iz+2);

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
__device__ inline void _load_1X00(const uint_t dummy, const uint_t iy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR00, ix0+0, iy, iz);
    r[texStart+1]  = tex3D(texR00, ix0+1, iy, iz);
    r[texStart+2]  = tex3D(texR00, ix0+2, iy, iz);
    r[texStart+3]  = tex3D(texR00, ix0+3, iy, iz);
    r[texStart+4]  = tex3D(texR00, ix0+4, iy, iz);

    u[texStart+0]  = tex3D(texU00, ix0+0, iy, iz);
    u[texStart+1]  = tex3D(texU00, ix0+1, iy, iz);
    u[texStart+2]  = tex3D(texU00, ix0+2, iy, iz);
    u[texStart+3]  = tex3D(texU00, ix0+3, iy, iz);
    u[texStart+4]  = tex3D(texU00, ix0+4, iy, iz);

    v[texStart+0]  = tex3D(texV00, ix0+0, iy, iz);
    v[texStart+1]  = tex3D(texV00, ix0+1, iy, iz);
    v[texStart+2]  = tex3D(texV00, ix0+2, iy, iz);
    v[texStart+3]  = tex3D(texV00, ix0+3, iy, iz);
    v[texStart+4]  = tex3D(texV00, ix0+4, iy, iz);

    w[texStart+0]  = tex3D(texW00, ix0+0, iy, iz);
    w[texStart+1]  = tex3D(texW00, ix0+1, iy, iz);
    w[texStart+2]  = tex3D(texW00, ix0+2, iy, iz);
    w[texStart+3]  = tex3D(texW00, ix0+3, iy, iz);
    w[texStart+4]  = tex3D(texW00, ix0+4, iy, iz);

    e[texStart+0]  = tex3D(texE00, ix0+0, iy, iz);
    e[texStart+1]  = tex3D(texE00, ix0+1, iy, iz);
    e[texStart+2]  = tex3D(texE00, ix0+2, iy, iz);
    e[texStart+3]  = tex3D(texE00, ix0+3, iy, iz);
    e[texStart+4]  = tex3D(texE00, ix0+4, iy, iz);

    G[texStart+0]  = tex3D(texG00, ix0+0, iy, iz);
    G[texStart+1]  = tex3D(texG00, ix0+1, iy, iz);
    G[texStart+2]  = tex3D(texG00, ix0+2, iy, iz);
    G[texStart+3]  = tex3D(texG00, ix0+3, iy, iz);
    G[texStart+4]  = tex3D(texG00, ix0+4, iy, iz);

    P[texStart+0]  = tex3D(texP00, ix0+0, iy, iz);
    P[texStart+1]  = tex3D(texP00, ix0+1, iy, iz);
    P[texStart+2]  = tex3D(texP00, ix0+2, iy, iz);
    P[texStart+3]  = tex3D(texP00, ix0+3, iy, iz);
    P[texStart+4]  = tex3D(texP00, ix0+4, iy, iz);

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
__device__ inline void _load_1Y00(const uint_t ix, const uint_t dummy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR00, ix, iy0+0, iz);
    r[texStart+1]  = tex3D(texR00, ix, iy0+1, iz);
    r[texStart+2]  = tex3D(texR00, ix, iy0+2, iz);
    r[texStart+3]  = tex3D(texR00, ix, iy0+3, iz);
    r[texStart+4]  = tex3D(texR00, ix, iy0+4, iz);

    u[texStart+0]  = tex3D(texU00, ix, iy0+0, iz);
    u[texStart+1]  = tex3D(texU00, ix, iy0+1, iz);
    u[texStart+2]  = tex3D(texU00, ix, iy0+2, iz);
    u[texStart+3]  = tex3D(texU00, ix, iy0+3, iz);
    u[texStart+4]  = tex3D(texU00, ix, iy0+4, iz);

    v[texStart+0]  = tex3D(texV00, ix, iy0+0, iz);
    v[texStart+1]  = tex3D(texV00, ix, iy0+1, iz);
    v[texStart+2]  = tex3D(texV00, ix, iy0+2, iz);
    v[texStart+3]  = tex3D(texV00, ix, iy0+3, iz);
    v[texStart+4]  = tex3D(texV00, ix, iy0+4, iz);

    w[texStart+0]  = tex3D(texW00, ix, iy0+0, iz);
    w[texStart+1]  = tex3D(texW00, ix, iy0+1, iz);
    w[texStart+2]  = tex3D(texW00, ix, iy0+2, iz);
    w[texStart+3]  = tex3D(texW00, ix, iy0+3, iz);
    w[texStart+4]  = tex3D(texW00, ix, iy0+4, iz);

    e[texStart+0]  = tex3D(texE00, ix, iy0+0, iz);
    e[texStart+1]  = tex3D(texE00, ix, iy0+1, iz);
    e[texStart+2]  = tex3D(texE00, ix, iy0+2, iz);
    e[texStart+3]  = tex3D(texE00, ix, iy0+3, iz);
    e[texStart+4]  = tex3D(texE00, ix, iy0+4, iz);

    G[texStart+0]  = tex3D(texG00, ix, iy0+0, iz);
    G[texStart+1]  = tex3D(texG00, ix, iy0+1, iz);
    G[texStart+2]  = tex3D(texG00, ix, iy0+2, iz);
    G[texStart+3]  = tex3D(texG00, ix, iy0+3, iz);
    G[texStart+4]  = tex3D(texG00, ix, iy0+4, iz);

    P[texStart+0]  = tex3D(texP00, ix, iy0+0, iz);
    P[texStart+1]  = tex3D(texP00, ix, iy0+1, iz);
    P[texStart+2]  = tex3D(texP00, ix, iy0+2, iz);
    P[texStart+3]  = tex3D(texP00, ix, iy0+3, iz);
    P[texStart+4]  = tex3D(texP00, ix, iy0+4, iz);

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
__device__ inline void _load_2X00(const uint_t dummy, const uint_t iy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR00, ix0+0, iy, iz);
    r[texStart+1]  = tex3D(texR00, ix0+1, iy, iz);
    r[texStart+2]  = tex3D(texR00, ix0+2, iy, iz);
    r[texStart+3]  = tex3D(texR00, ix0+3, iy, iz);

    u[texStart+0]  = tex3D(texU00, ix0+0, iy, iz);
    u[texStart+1]  = tex3D(texU00, ix0+1, iy, iz);
    u[texStart+2]  = tex3D(texU00, ix0+2, iy, iz);
    u[texStart+3]  = tex3D(texU00, ix0+3, iy, iz);

    v[texStart+0]  = tex3D(texV00, ix0+0, iy, iz);
    v[texStart+1]  = tex3D(texV00, ix0+1, iy, iz);
    v[texStart+2]  = tex3D(texV00, ix0+2, iy, iz);
    v[texStart+3]  = tex3D(texV00, ix0+3, iy, iz);

    w[texStart+0]  = tex3D(texW00, ix0+0, iy, iz);
    w[texStart+1]  = tex3D(texW00, ix0+1, iy, iz);
    w[texStart+2]  = tex3D(texW00, ix0+2, iy, iz);
    w[texStart+3]  = tex3D(texW00, ix0+3, iy, iz);

    e[texStart+0]  = tex3D(texE00, ix0+0, iy, iz);
    e[texStart+1]  = tex3D(texE00, ix0+1, iy, iz);
    e[texStart+2]  = tex3D(texE00, ix0+2, iy, iz);
    e[texStart+3]  = tex3D(texE00, ix0+3, iy, iz);

    G[texStart+0]  = tex3D(texG00, ix0+0, iy, iz);
    G[texStart+1]  = tex3D(texG00, ix0+1, iy, iz);
    G[texStart+2]  = tex3D(texG00, ix0+2, iy, iz);
    G[texStart+3]  = tex3D(texG00, ix0+3, iy, iz);

    P[texStart+0]  = tex3D(texP00, ix0+0, iy, iz);
    P[texStart+1]  = tex3D(texP00, ix0+1, iy, iz);
    P[texStart+2]  = tex3D(texP00, ix0+2, iy, iz);
    P[texStart+3]  = tex3D(texP00, ix0+3, iy, iz);

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
__device__ inline void _load_2Y00(const uint_t ix, const uint_t dummy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR00, ix, iy0+0, iz);
    r[texStart+1]  = tex3D(texR00, ix, iy0+1, iz);
    r[texStart+2]  = tex3D(texR00, ix, iy0+2, iz);
    r[texStart+3]  = tex3D(texR00, ix, iy0+3, iz);

    u[texStart+0]  = tex3D(texU00, ix, iy0+0, iz);
    u[texStart+1]  = tex3D(texU00, ix, iy0+1, iz);
    u[texStart+2]  = tex3D(texU00, ix, iy0+2, iz);
    u[texStart+3]  = tex3D(texU00, ix, iy0+3, iz);

    v[texStart+0]  = tex3D(texV00, ix, iy0+0, iz);
    v[texStart+1]  = tex3D(texV00, ix, iy0+1, iz);
    v[texStart+2]  = tex3D(texV00, ix, iy0+2, iz);
    v[texStart+3]  = tex3D(texV00, ix, iy0+3, iz);

    w[texStart+0]  = tex3D(texW00, ix, iy0+0, iz);
    w[texStart+1]  = tex3D(texW00, ix, iy0+1, iz);
    w[texStart+2]  = tex3D(texW00, ix, iy0+2, iz);
    w[texStart+3]  = tex3D(texW00, ix, iy0+3, iz);

    e[texStart+0]  = tex3D(texE00, ix, iy0+0, iz);
    e[texStart+1]  = tex3D(texE00, ix, iy0+1, iz);
    e[texStart+2]  = tex3D(texE00, ix, iy0+2, iz);
    e[texStart+3]  = tex3D(texE00, ix, iy0+3, iz);

    G[texStart+0]  = tex3D(texG00, ix, iy0+0, iz);
    G[texStart+1]  = tex3D(texG00, ix, iy0+1, iz);
    G[texStart+2]  = tex3D(texG00, ix, iy0+2, iz);
    G[texStart+3]  = tex3D(texG00, ix, iy0+3, iz);

    P[texStart+0]  = tex3D(texP00, ix, iy0+0, iz);
    P[texStart+1]  = tex3D(texP00, ix, iy0+1, iz);
    P[texStart+2]  = tex3D(texP00, ix, iy0+2, iz);
    P[texStart+3]  = tex3D(texP00, ix, iy0+3, iz);

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
__device__ inline void _load_3X00(const uint_t dummy, const uint_t iy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR00, ix0+0, iy, iz);
    r[texStart+1]  = tex3D(texR00, ix0+1, iy, iz);
    r[texStart+2]  = tex3D(texR00, ix0+2, iy, iz);

    u[texStart+0]  = tex3D(texU00, ix0+0, iy, iz);
    u[texStart+1]  = tex3D(texU00, ix0+1, iy, iz);
    u[texStart+2]  = tex3D(texU00, ix0+2, iy, iz);

    v[texStart+0]  = tex3D(texV00, ix0+0, iy, iz);
    v[texStart+1]  = tex3D(texV00, ix0+1, iy, iz);
    v[texStart+2]  = tex3D(texV00, ix0+2, iy, iz);

    w[texStart+0]  = tex3D(texW00, ix0+0, iy, iz);
    w[texStart+1]  = tex3D(texW00, ix0+1, iy, iz);
    w[texStart+2]  = tex3D(texW00, ix0+2, iy, iz);

    e[texStart+0]  = tex3D(texE00, ix0+0, iy, iz);
    e[texStart+1]  = tex3D(texE00, ix0+1, iy, iz);
    e[texStart+2]  = tex3D(texE00, ix0+2, iy, iz);

    G[texStart+0]  = tex3D(texG00, ix0+0, iy, iz);
    G[texStart+1]  = tex3D(texG00, ix0+1, iy, iz);
    G[texStart+2]  = tex3D(texG00, ix0+2, iy, iz);

    P[texStart+0]  = tex3D(texP00, ix0+0, iy, iz);
    P[texStart+1]  = tex3D(texP00, ix0+1, iy, iz);
    P[texStart+2]  = tex3D(texP00, ix0+2, iy, iz);

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
__device__ inline void _load_3Y00(const uint_t ix, const uint_t dummy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR00, ix, iy0+0, iz);
    r[texStart+1]  = tex3D(texR00, ix, iy0+1, iz);
    r[texStart+2]  = tex3D(texR00, ix, iy0+2, iz);

    u[texStart+0]  = tex3D(texU00, ix, iy0+0, iz);
    u[texStart+1]  = tex3D(texU00, ix, iy0+1, iz);
    u[texStart+2]  = tex3D(texU00, ix, iy0+2, iz);

    v[texStart+0]  = tex3D(texV00, ix, iy0+0, iz);
    v[texStart+1]  = tex3D(texV00, ix, iy0+1, iz);
    v[texStart+2]  = tex3D(texV00, ix, iy0+2, iz);

    w[texStart+0]  = tex3D(texW00, ix, iy0+0, iz);
    w[texStart+1]  = tex3D(texW00, ix, iy0+1, iz);
    w[texStart+2]  = tex3D(texW00, ix, iy0+2, iz);

    e[texStart+0]  = tex3D(texE00, ix, iy0+0, iz);
    e[texStart+1]  = tex3D(texE00, ix, iy0+1, iz);
    e[texStart+2]  = tex3D(texE00, ix, iy0+2, iz);

    G[texStart+0]  = tex3D(texG00, ix, iy0+0, iz);
    G[texStart+1]  = tex3D(texG00, ix, iy0+1, iz);
    G[texStart+2]  = tex3D(texG00, ix, iy0+2, iz);

    P[texStart+0]  = tex3D(texP00, ix, iy0+0, iz);
    P[texStart+1]  = tex3D(texP00, ix, iy0+1, iz);
    P[texStart+2]  = tex3D(texP00, ix, iy0+2, iz);

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

///////////////////////////////////////////////////////////////////////////////
// Reading from 01
__device__
inline void _load_internal_X01(const uint_t ix, const uint_t iy, const uint_t iz,
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
    r[0]  = tex3D(texR01, ix-3, iy, iz);
    r[1]  = tex3D(texR01, ix-2, iy, iz);
    r[2]  = tex3D(texR01, ix-1, iy, iz);
    r[3]  = tex3D(texR01, ix,   iy, iz);
    r[4]  = tex3D(texR01, ix+1, iy, iz);
    r[5]  = tex3D(texR01, ix+2, iy, iz);

    u[0]  = tex3D(texU01, ix-3, iy, iz);
    u[1]  = tex3D(texU01, ix-2, iy, iz);
    u[2]  = tex3D(texU01, ix-1, iy, iz);
    u[3]  = tex3D(texU01, ix,   iy, iz);
    u[4]  = tex3D(texU01, ix+1, iy, iz);
    u[5]  = tex3D(texU01, ix+2, iy, iz);

    v[0]  = tex3D(texV01, ix-3, iy, iz);
    v[1]  = tex3D(texV01, ix-2, iy, iz);
    v[2]  = tex3D(texV01, ix-1, iy, iz);
    v[3]  = tex3D(texV01, ix,   iy, iz);
    v[4]  = tex3D(texV01, ix+1, iy, iz);
    v[5]  = tex3D(texV01, ix+2, iy, iz);

    w[0]  = tex3D(texW01, ix-3, iy, iz);
    w[1]  = tex3D(texW01, ix-2, iy, iz);
    w[2]  = tex3D(texW01, ix-1, iy, iz);
    w[3]  = tex3D(texW01, ix,   iy, iz);
    w[4]  = tex3D(texW01, ix+1, iy, iz);
    w[5]  = tex3D(texW01, ix+2, iy, iz);

    e[0]  = tex3D(texE01, ix-3, iy, iz);
    e[1]  = tex3D(texE01, ix-2, iy, iz);
    e[2]  = tex3D(texE01, ix-1, iy, iz);
    e[3]  = tex3D(texE01, ix,   iy, iz);
    e[4]  = tex3D(texE01, ix+1, iy, iz);
    e[5]  = tex3D(texE01, ix+2, iy, iz);

    G[0]  = tex3D(texG01, ix-3, iy, iz);
    G[1]  = tex3D(texG01, ix-2, iy, iz);
    G[2]  = tex3D(texG01, ix-1, iy, iz);
    G[3]  = tex3D(texG01, ix,   iy, iz);
    G[4]  = tex3D(texG01, ix+1, iy, iz);
    G[5]  = tex3D(texG01, ix+2, iy, iz);

    P[0]  = tex3D(texP01, ix-3, iy, iz);
    P[1]  = tex3D(texP01, ix-2, iy, iz);
    P[2]  = tex3D(texP01, ix-1, iy, iz);
    P[3]  = tex3D(texP01, ix,   iy, iz);
    P[4]  = tex3D(texP01, ix+1, iy, iz);
    P[5]  = tex3D(texP01, ix+2, iy, iz);

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
inline void _load_internal_Y01(const uint_t ix, const uint_t iy, const uint_t iz,
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
    r[0]  = tex3D(texR01, ix, iy-3, iz);
    r[1]  = tex3D(texR01, ix, iy-2, iz);
    r[2]  = tex3D(texR01, ix, iy-1, iz);
    r[3]  = tex3D(texR01, ix, iy,   iz);
    r[4]  = tex3D(texR01, ix, iy+1, iz);
    r[5]  = tex3D(texR01, ix, iy+2, iz);

    u[0]  = tex3D(texU01, ix, iy-3, iz);
    u[1]  = tex3D(texU01, ix, iy-2, iz);
    u[2]  = tex3D(texU01, ix, iy-1, iz);
    u[3]  = tex3D(texU01, ix, iy,   iz);
    u[4]  = tex3D(texU01, ix, iy+1, iz);
    u[5]  = tex3D(texU01, ix, iy+2, iz);

    v[0]  = tex3D(texV01, ix, iy-3, iz);
    v[1]  = tex3D(texV01, ix, iy-2, iz);
    v[2]  = tex3D(texV01, ix, iy-1, iz);
    v[3]  = tex3D(texV01, ix, iy,   iz);
    v[4]  = tex3D(texV01, ix, iy+1, iz);
    v[5]  = tex3D(texV01, ix, iy+2, iz);

    w[0]  = tex3D(texW01, ix, iy-3, iz);
    w[1]  = tex3D(texW01, ix, iy-2, iz);
    w[2]  = tex3D(texW01, ix, iy-1, iz);
    w[3]  = tex3D(texW01, ix, iy,   iz);
    w[4]  = tex3D(texW01, ix, iy+1, iz);
    w[5]  = tex3D(texW01, ix, iy+2, iz);

    e[0]  = tex3D(texE01, ix, iy-3, iz);
    e[1]  = tex3D(texE01, ix, iy-2, iz);
    e[2]  = tex3D(texE01, ix, iy-1, iz);
    e[3]  = tex3D(texE01, ix, iy,   iz);
    e[4]  = tex3D(texE01, ix, iy+1, iz);
    e[5]  = tex3D(texE01, ix, iy+2, iz);

    G[0]  = tex3D(texG01, ix, iy-3, iz);
    G[1]  = tex3D(texG01, ix, iy-2, iz);
    G[2]  = tex3D(texG01, ix, iy-1, iz);
    G[3]  = tex3D(texG01, ix, iy,   iz);
    G[4]  = tex3D(texG01, ix, iy+1, iz);
    G[5]  = tex3D(texG01, ix, iy+2, iz);

    P[0]  = tex3D(texP01, ix, iy-3, iz);
    P[1]  = tex3D(texP01, ix, iy-2, iz);
    P[2]  = tex3D(texP01, ix, iy-1, iz);
    P[3]  = tex3D(texP01, ix, iy,   iz);
    P[4]  = tex3D(texP01, ix, iy+1, iz);
    P[5]  = tex3D(texP01, ix, iy+2, iz);

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
inline void _load_internal_Z01(const uint_t ix, const uint_t iy, const uint_t iz,
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
    r[0]  = tex3D(texR01, ix, iy, iz-3);
    r[1]  = tex3D(texR01, ix, iy, iz-2);
    r[2]  = tex3D(texR01, ix, iy, iz-1);
    r[3]  = tex3D(texR01, ix, iy, iz);
    r[4]  = tex3D(texR01, ix, iy, iz+1);
    r[5]  = tex3D(texR01, ix, iy, iz+2);

    u[0]  = tex3D(texU01, ix, iy, iz-3);
    u[1]  = tex3D(texU01, ix, iy, iz-2);
    u[2]  = tex3D(texU01, ix, iy, iz-1);
    u[3]  = tex3D(texU01, ix, iy, iz);
    u[4]  = tex3D(texU01, ix, iy, iz+1);
    u[5]  = tex3D(texU01, ix, iy, iz+2);

    v[0]  = tex3D(texV01, ix, iy, iz-3);
    v[1]  = tex3D(texV01, ix, iy, iz-2);
    v[2]  = tex3D(texV01, ix, iy, iz-1);
    v[3]  = tex3D(texV01, ix, iy, iz);
    v[4]  = tex3D(texV01, ix, iy, iz+1);
    v[5]  = tex3D(texV01, ix, iy, iz+2);

    w[0]  = tex3D(texW01, ix, iy, iz-3);
    w[1]  = tex3D(texW01, ix, iy, iz-2);
    w[2]  = tex3D(texW01, ix, iy, iz-1);
    w[3]  = tex3D(texW01, ix, iy, iz);
    w[4]  = tex3D(texW01, ix, iy, iz+1);
    w[5]  = tex3D(texW01, ix, iy, iz+2);

    e[0]  = tex3D(texE01, ix, iy, iz-3);
    e[1]  = tex3D(texE01, ix, iy, iz-2);
    e[2]  = tex3D(texE01, ix, iy, iz-1);
    e[3]  = tex3D(texE01, ix, iy, iz);
    e[4]  = tex3D(texE01, ix, iy, iz+1);
    e[5]  = tex3D(texE01, ix, iy, iz+2);

    G[0]  = tex3D(texG01, ix, iy, iz-3);
    G[1]  = tex3D(texG01, ix, iy, iz-2);
    G[2]  = tex3D(texG01, ix, iy, iz-1);
    G[3]  = tex3D(texG01, ix, iy, iz);
    G[4]  = tex3D(texG01, ix, iy, iz+1);
    G[5]  = tex3D(texG01, ix, iy, iz+2);

    P[0]  = tex3D(texP01, ix, iy, iz-3);
    P[1]  = tex3D(texP01, ix, iy, iz-2);
    P[2]  = tex3D(texP01, ix, iy, iz-1);
    P[3]  = tex3D(texP01, ix, iy, iz);
    P[4]  = tex3D(texP01, ix, iy, iz+1);
    P[5]  = tex3D(texP01, ix, iy, iz+2);

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
__device__ inline void _load_1X01(const uint_t dummy, const uint_t iy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR01, ix0+0, iy, iz);
    r[texStart+1]  = tex3D(texR01, ix0+1, iy, iz);
    r[texStart+2]  = tex3D(texR01, ix0+2, iy, iz);
    r[texStart+3]  = tex3D(texR01, ix0+3, iy, iz);
    r[texStart+4]  = tex3D(texR01, ix0+4, iy, iz);

    u[texStart+0]  = tex3D(texU01, ix0+0, iy, iz);
    u[texStart+1]  = tex3D(texU01, ix0+1, iy, iz);
    u[texStart+2]  = tex3D(texU01, ix0+2, iy, iz);
    u[texStart+3]  = tex3D(texU01, ix0+3, iy, iz);
    u[texStart+4]  = tex3D(texU01, ix0+4, iy, iz);

    v[texStart+0]  = tex3D(texV01, ix0+0, iy, iz);
    v[texStart+1]  = tex3D(texV01, ix0+1, iy, iz);
    v[texStart+2]  = tex3D(texV01, ix0+2, iy, iz);
    v[texStart+3]  = tex3D(texV01, ix0+3, iy, iz);
    v[texStart+4]  = tex3D(texV01, ix0+4, iy, iz);

    w[texStart+0]  = tex3D(texW01, ix0+0, iy, iz);
    w[texStart+1]  = tex3D(texW01, ix0+1, iy, iz);
    w[texStart+2]  = tex3D(texW01, ix0+2, iy, iz);
    w[texStart+3]  = tex3D(texW01, ix0+3, iy, iz);
    w[texStart+4]  = tex3D(texW01, ix0+4, iy, iz);

    e[texStart+0]  = tex3D(texE01, ix0+0, iy, iz);
    e[texStart+1]  = tex3D(texE01, ix0+1, iy, iz);
    e[texStart+2]  = tex3D(texE01, ix0+2, iy, iz);
    e[texStart+3]  = tex3D(texE01, ix0+3, iy, iz);
    e[texStart+4]  = tex3D(texE01, ix0+4, iy, iz);

    G[texStart+0]  = tex3D(texG01, ix0+0, iy, iz);
    G[texStart+1]  = tex3D(texG01, ix0+1, iy, iz);
    G[texStart+2]  = tex3D(texG01, ix0+2, iy, iz);
    G[texStart+3]  = tex3D(texG01, ix0+3, iy, iz);
    G[texStart+4]  = tex3D(texG01, ix0+4, iy, iz);

    P[texStart+0]  = tex3D(texP01, ix0+0, iy, iz);
    P[texStart+1]  = tex3D(texP01, ix0+1, iy, iz);
    P[texStart+2]  = tex3D(texP01, ix0+2, iy, iz);
    P[texStart+3]  = tex3D(texP01, ix0+3, iy, iz);
    P[texStart+4]  = tex3D(texP01, ix0+4, iy, iz);

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
__device__ inline void _load_1Y01(const uint_t ix, const uint_t dummy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR01, ix, iy0+0, iz);
    r[texStart+1]  = tex3D(texR01, ix, iy0+1, iz);
    r[texStart+2]  = tex3D(texR01, ix, iy0+2, iz);
    r[texStart+3]  = tex3D(texR01, ix, iy0+3, iz);
    r[texStart+4]  = tex3D(texR01, ix, iy0+4, iz);

    u[texStart+0]  = tex3D(texU01, ix, iy0+0, iz);
    u[texStart+1]  = tex3D(texU01, ix, iy0+1, iz);
    u[texStart+2]  = tex3D(texU01, ix, iy0+2, iz);
    u[texStart+3]  = tex3D(texU01, ix, iy0+3, iz);
    u[texStart+4]  = tex3D(texU01, ix, iy0+4, iz);

    v[texStart+0]  = tex3D(texV01, ix, iy0+0, iz);
    v[texStart+1]  = tex3D(texV01, ix, iy0+1, iz);
    v[texStart+2]  = tex3D(texV01, ix, iy0+2, iz);
    v[texStart+3]  = tex3D(texV01, ix, iy0+3, iz);
    v[texStart+4]  = tex3D(texV01, ix, iy0+4, iz);

    w[texStart+0]  = tex3D(texW01, ix, iy0+0, iz);
    w[texStart+1]  = tex3D(texW01, ix, iy0+1, iz);
    w[texStart+2]  = tex3D(texW01, ix, iy0+2, iz);
    w[texStart+3]  = tex3D(texW01, ix, iy0+3, iz);
    w[texStart+4]  = tex3D(texW01, ix, iy0+4, iz);

    e[texStart+0]  = tex3D(texE01, ix, iy0+0, iz);
    e[texStart+1]  = tex3D(texE01, ix, iy0+1, iz);
    e[texStart+2]  = tex3D(texE01, ix, iy0+2, iz);
    e[texStart+3]  = tex3D(texE01, ix, iy0+3, iz);
    e[texStart+4]  = tex3D(texE01, ix, iy0+4, iz);

    G[texStart+0]  = tex3D(texG01, ix, iy0+0, iz);
    G[texStart+1]  = tex3D(texG01, ix, iy0+1, iz);
    G[texStart+2]  = tex3D(texG01, ix, iy0+2, iz);
    G[texStart+3]  = tex3D(texG01, ix, iy0+3, iz);
    G[texStart+4]  = tex3D(texG01, ix, iy0+4, iz);

    P[texStart+0]  = tex3D(texP01, ix, iy0+0, iz);
    P[texStart+1]  = tex3D(texP01, ix, iy0+1, iz);
    P[texStart+2]  = tex3D(texP01, ix, iy0+2, iz);
    P[texStart+3]  = tex3D(texP01, ix, iy0+3, iz);
    P[texStart+4]  = tex3D(texP01, ix, iy0+4, iz);

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
__device__ inline void _load_2X01(const uint_t dummy, const uint_t iy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR01, ix0+0, iy, iz);
    r[texStart+1]  = tex3D(texR01, ix0+1, iy, iz);
    r[texStart+2]  = tex3D(texR01, ix0+2, iy, iz);
    r[texStart+3]  = tex3D(texR01, ix0+3, iy, iz);

    u[texStart+0]  = tex3D(texU01, ix0+0, iy, iz);
    u[texStart+1]  = tex3D(texU01, ix0+1, iy, iz);
    u[texStart+2]  = tex3D(texU01, ix0+2, iy, iz);
    u[texStart+3]  = tex3D(texU01, ix0+3, iy, iz);

    v[texStart+0]  = tex3D(texV01, ix0+0, iy, iz);
    v[texStart+1]  = tex3D(texV01, ix0+1, iy, iz);
    v[texStart+2]  = tex3D(texV01, ix0+2, iy, iz);
    v[texStart+3]  = tex3D(texV01, ix0+3, iy, iz);

    w[texStart+0]  = tex3D(texW01, ix0+0, iy, iz);
    w[texStart+1]  = tex3D(texW01, ix0+1, iy, iz);
    w[texStart+2]  = tex3D(texW01, ix0+2, iy, iz);
    w[texStart+3]  = tex3D(texW01, ix0+3, iy, iz);

    e[texStart+0]  = tex3D(texE01, ix0+0, iy, iz);
    e[texStart+1]  = tex3D(texE01, ix0+1, iy, iz);
    e[texStart+2]  = tex3D(texE01, ix0+2, iy, iz);
    e[texStart+3]  = tex3D(texE01, ix0+3, iy, iz);

    G[texStart+0]  = tex3D(texG01, ix0+0, iy, iz);
    G[texStart+1]  = tex3D(texG01, ix0+1, iy, iz);
    G[texStart+2]  = tex3D(texG01, ix0+2, iy, iz);
    G[texStart+3]  = tex3D(texG01, ix0+3, iy, iz);

    P[texStart+0]  = tex3D(texP01, ix0+0, iy, iz);
    P[texStart+1]  = tex3D(texP01, ix0+1, iy, iz);
    P[texStart+2]  = tex3D(texP01, ix0+2, iy, iz);
    P[texStart+3]  = tex3D(texP01, ix0+3, iy, iz);

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
__device__ inline void _load_2Y01(const uint_t ix, const uint_t dummy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR01, ix, iy0+0, iz);
    r[texStart+1]  = tex3D(texR01, ix, iy0+1, iz);
    r[texStart+2]  = tex3D(texR01, ix, iy0+2, iz);
    r[texStart+3]  = tex3D(texR01, ix, iy0+3, iz);

    u[texStart+0]  = tex3D(texU01, ix, iy0+0, iz);
    u[texStart+1]  = tex3D(texU01, ix, iy0+1, iz);
    u[texStart+2]  = tex3D(texU01, ix, iy0+2, iz);
    u[texStart+3]  = tex3D(texU01, ix, iy0+3, iz);

    v[texStart+0]  = tex3D(texV01, ix, iy0+0, iz);
    v[texStart+1]  = tex3D(texV01, ix, iy0+1, iz);
    v[texStart+2]  = tex3D(texV01, ix, iy0+2, iz);
    v[texStart+3]  = tex3D(texV01, ix, iy0+3, iz);

    w[texStart+0]  = tex3D(texW01, ix, iy0+0, iz);
    w[texStart+1]  = tex3D(texW01, ix, iy0+1, iz);
    w[texStart+2]  = tex3D(texW01, ix, iy0+2, iz);
    w[texStart+3]  = tex3D(texW01, ix, iy0+3, iz);

    e[texStart+0]  = tex3D(texE01, ix, iy0+0, iz);
    e[texStart+1]  = tex3D(texE01, ix, iy0+1, iz);
    e[texStart+2]  = tex3D(texE01, ix, iy0+2, iz);
    e[texStart+3]  = tex3D(texE01, ix, iy0+3, iz);

    G[texStart+0]  = tex3D(texG01, ix, iy0+0, iz);
    G[texStart+1]  = tex3D(texG01, ix, iy0+1, iz);
    G[texStart+2]  = tex3D(texG01, ix, iy0+2, iz);
    G[texStart+3]  = tex3D(texG01, ix, iy0+3, iz);

    P[texStart+0]  = tex3D(texP01, ix, iy0+0, iz);
    P[texStart+1]  = tex3D(texP01, ix, iy0+1, iz);
    P[texStart+2]  = tex3D(texP01, ix, iy0+2, iz);
    P[texStart+3]  = tex3D(texP01, ix, iy0+3, iz);

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
__device__ inline void _load_3X01(const uint_t dummy, const uint_t iy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR01, ix0+0, iy, iz);
    r[texStart+1]  = tex3D(texR01, ix0+1, iy, iz);
    r[texStart+2]  = tex3D(texR01, ix0+2, iy, iz);

    u[texStart+0]  = tex3D(texU01, ix0+0, iy, iz);
    u[texStart+1]  = tex3D(texU01, ix0+1, iy, iz);
    u[texStart+2]  = tex3D(texU01, ix0+2, iy, iz);

    v[texStart+0]  = tex3D(texV01, ix0+0, iy, iz);
    v[texStart+1]  = tex3D(texV01, ix0+1, iy, iz);
    v[texStart+2]  = tex3D(texV01, ix0+2, iy, iz);

    w[texStart+0]  = tex3D(texW01, ix0+0, iy, iz);
    w[texStart+1]  = tex3D(texW01, ix0+1, iy, iz);
    w[texStart+2]  = tex3D(texW01, ix0+2, iy, iz);

    e[texStart+0]  = tex3D(texE01, ix0+0, iy, iz);
    e[texStart+1]  = tex3D(texE01, ix0+1, iy, iz);
    e[texStart+2]  = tex3D(texE01, ix0+2, iy, iz);

    G[texStart+0]  = tex3D(texG01, ix0+0, iy, iz);
    G[texStart+1]  = tex3D(texG01, ix0+1, iy, iz);
    G[texStart+2]  = tex3D(texG01, ix0+2, iy, iz);

    P[texStart+0]  = tex3D(texP01, ix0+0, iy, iz);
    P[texStart+1]  = tex3D(texP01, ix0+1, iy, iz);
    P[texStart+2]  = tex3D(texP01, ix0+2, iy, iz);

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
__device__ inline void _load_3Y01(const uint_t ix, const uint_t dummy, const uint_t iz,
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
    r[texStart+0]  = tex3D(texR01, ix, iy0+0, iz);
    r[texStart+1]  = tex3D(texR01, ix, iy0+1, iz);
    r[texStart+2]  = tex3D(texR01, ix, iy0+2, iz);

    u[texStart+0]  = tex3D(texU01, ix, iy0+0, iz);
    u[texStart+1]  = tex3D(texU01, ix, iy0+1, iz);
    u[texStart+2]  = tex3D(texU01, ix, iy0+2, iz);

    v[texStart+0]  = tex3D(texV01, ix, iy0+0, iz);
    v[texStart+1]  = tex3D(texV01, ix, iy0+1, iz);
    v[texStart+2]  = tex3D(texV01, ix, iy0+2, iz);

    w[texStart+0]  = tex3D(texW01, ix, iy0+0, iz);
    w[texStart+1]  = tex3D(texW01, ix, iy0+1, iz);
    w[texStart+2]  = tex3D(texW01, ix, iy0+2, iz);

    e[texStart+0]  = tex3D(texE01, ix, iy0+0, iz);
    e[texStart+1]  = tex3D(texE01, ix, iy0+1, iz);
    e[texStart+2]  = tex3D(texE01, ix, iy0+2, iz);

    G[texStart+0]  = tex3D(texG01, ix, iy0+0, iz);
    G[texStart+1]  = tex3D(texG01, ix, iy0+1, iz);
    G[texStart+2]  = tex3D(texG01, ix, iy0+2, iz);

    P[texStart+0]  = tex3D(texP01, ix, iy0+0, iz);
    P[texStart+1]  = tex3D(texP01, ix, iy0+1, iz);
    P[texStart+2]  = tex3D(texP01, ix, iy0+2, iz);

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

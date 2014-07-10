/* *
 * Types.h
 *
 * Created by Fabian Wermelinger on 6/24/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include <stdio.h>
#include <assert.h>
#include <vector>

#include "NodeBlock.h"

#ifdef _FLOAT_PRECISION_
typedef float Real;
#else
typedef double Real;
#endif

typedef std::vector<Real *> RealPtrVec_t;
typedef unsigned int uint_t;


///////////////////////////////////////////////////////////////////////////////
// INDEX MAPPINGS USED FOR HOST AND DEVICE GHOST BUFFERS
///////////////////////////////////////////////////////////////////////////////
#define ID3(ix,iy,iz,NX,NY) ((ix) + (NX) * ((iy) + (NY) * (iz)))
/* *
 * Actual mappings into ghost buffers.  Macros are used here because these
 * mappings are used in host code as well as device code.  In C-notation, the
 * mappings defined below are used for bounds:
 *
 * GHOSTX[NodeBlock::sizeZ][3][NodeBlock::sizeY]
 * GHOSTY[NodeBlock::sizeZ][NodeBlock::sizeX][3]
 *
 * The layout is chosen such that a warp access is coalesced
 * */
#define GHOSTMAPX(ix,iy,iz) ((iy) + NodeBlock::sizeY * ((ix) + 3 * (iz)))
#define GHOSTMAPY(ix,iy,iz) ((iy) + 3 * ((ix) + NodeBlock::sizeX * (iz)))

extern "C"
{
    typedef uint_t (*index_map)(const int, const int, const int);

    struct ghostmap
    {
        // Used for host code (more convenient and less error prone)
        static inline uint_t X(const int ix, const int iy, const int iz) { return GHOSTMAPX(ix,iy,iz); }
        static inline uint_t Y(const int ix, const int iy, const int iz) { return GHOSTMAPY(ix,iy,iz); }
        static inline uint_t Z(const int ix, const int iy, const int iz) { return ID3(ix, iy, iz, NodeBlock::sizeX, NodeBlock::sizeY); }
    };

    struct flesh2ghost
    {
        /* *
         * These indexers define offsets used to fill the GPU ghost buffers
         * according to the mapping defined in ghostmap when extracting halo
         * cells from the internal grid (flesh).  These buffers are
         * communicated with MPI.
         * A note on left (L) and right (R): say coordinate q1 < q2 in some
         * frame of reference, then q1 corresponds to L and q2 to R.
         * */
        static inline uint_t X_L(const int ix, const int iy, const int iz) { return ghostmap::X(ix, iy, iz); }
        static inline uint_t X_R(const int ix, const int iy, const int iz) { return ghostmap::X(ix-NodeBlock::sizeX+3, iy, iz); }
        static inline uint_t Y_L(const int ix, const int iy, const int iz) { return ghostmap::Y(ix, iy, iz); }
        static inline uint_t Y_R(const int ix, const int iy, const int iz) { return ghostmap::Y(ix, iy-NodeBlock::sizeY+3, iz); }
    };
}


///////////////////////////////////////////////////////////////////////////////
// OUTPUT STREAMER
///////////////////////////////////////////////////////////////////////////////
struct StreamerGridPointIterative //dummy
{
    static const int channels = 7;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    inline int _id(const int ix, const int iy, const int iz) const { assert(ix + NX * (iy + NY * iz) < NX*NY*NZ); return ix + NX * (iy + NY * iz); }

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    StreamerGridPointIterative() : r(NULL), u(NULL), v(NULL), w(NULL), e(NULL), G(NULL), P(NULL) {} // can not access data, only name().. not a clean solution!
    StreamerGridPointIterative(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    template<int channel>
    inline Real operate(const int ix, const int iy, const int iz) { abort(); return 0; }

    const char * name() { return "StreamerGridPointIterative" ; }
};

template<> inline Real StreamerGridPointIterative::operate<0>(const int ix, const int iy, const int iz) { return r[_id(ix,iy,iz)]; }
template<> inline Real StreamerGridPointIterative::operate<1>(const int ix, const int iy, const int iz) { const int idx = _id(ix,iy,iz); return u[idx]/r[idx]; }
template<> inline Real StreamerGridPointIterative::operate<2>(const int ix, const int iy, const int iz) { const int idx = _id(ix,iy,iz); return v[idx]/r[idx]; }
template<> inline Real StreamerGridPointIterative::operate<3>(const int ix, const int iy, const int iz) { const int idx = _id(ix,iy,iz); return w[idx]/r[idx]; }
template<> inline Real StreamerGridPointIterative::operate<4>(const int ix, const int iy, const int iz) { const int idx = _id(ix,iy,iz); return (e[idx]-0.5*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx])/r[idx] - P[idx])/G[idx]; }
template<> inline Real StreamerGridPointIterative::operate<5>(const int ix, const int iy, const int iz) { return G[_id(ix,iy,iz)]; }
template<> inline Real StreamerGridPointIterative::operate<6>(const int ix, const int iy, const int iz) { return P[_id(ix,iy,iz)]; }


struct myTensorialStreamer
{
    static const int NCHANNELS = 9;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    inline int _id(const int ix, const int iy, const int iz) const { assert(ix + NX * (iy + NY * iz) < NX*NY*NZ); return ix + NX * (iy + NY * iz); }

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    myTensorialStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = _id(ix,iy,iz);
        assert(idx < NX * NY * NZ);
        out[0] = r[idx];
        out[1] = u[idx]/r[idx];
        out[2] = v[idx]/r[idx];
        out[3] = w[idx]/r[idx];
        out[4] = (e[idx]-0.5*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx])/r[idx] - P[idx])/G[idx];
        out[5] = G[idx];
        out[6] = P[idx];
        out[7] = 0.;
        out[8] = 0.;
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz) const
    {
        const int idx = _id(ix,iy,iz);
        assert(idx < NX * NY * NZ);
        // dummy
    }

    static const char * getAttributeName() { return "Tensor"; }
};


struct myScalarStreamer
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    inline int _id(const int ix, const int iy, const int iz) const { assert(ix + NX * (iy + NY * iz) < NX*NY*NZ); return ix + NX * (iy + NY * iz); }

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    myScalarStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = _id(ix,iy,iz);
        assert(idx < NX * NY * NZ);
        out[0] = r[idx];
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz) const
    {
        const int idx = _id(ix,iy,iz);
        assert(idx < NX * NY * NZ);
        // dummy
    }

    static const char * getAttributeName() { return "Scalar"; }
};

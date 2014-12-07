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
#include <cstdlib>
#include <algorithm>

#include "NodeBlock.h"

#ifdef _FLOAT_PRECISION_
typedef float Real;
#else
typedef double Real;
#endif

typedef std::vector<Real *> real_vector_t; // just a vector holding pointers
typedef unsigned int uint_t;

enum Coord {X=0, Y, Z};

struct FluidElement
{
    Real rho, u, v, w, energy, G, P;
    FluidElement() : rho(-1), u(0), v(0), w(0), energy(-1), G(-1), P(-1) { }
    FluidElement(const FluidElement& e): rho(e.rho), u(e.u), v(e.v), w(e.w), energy(e.energy), G(e.G), P(e.P) { }
    FluidElement& operator=(const FluidElement& e)
    {
        if (this != &e) {rho = e.rho; u = e.u; v = e.v; w = e.w; energy = e.energy; G = e.G; P = e.P;}
        return *this;
    }
    FluidElement operator+(const FluidElement& e) const
    {
        FluidElement sum(e);
        sum.rho += rho; sum.u += u; sum.v += v; sum.w += w; sum.energy += energy; sum.G += G; sum.P += P;
        return sum;
    }
    friend FluidElement operator*(const Real s, const FluidElement& e);
};

FluidElement operator*(const Real s, const FluidElement& e);


///////////////////////////////////////////////////////////////////////////////
// INDEX MAPPINGS USED FOR HOST AND DEVICE GHOST BUFFERS
///////////////////////////////////////////////////////////////////////////////
#define ID3(ix,iy,iz,NX,NY) ((ix) + (NX) * ((iy) + (NY) * (iz)))
/* *
 * Actual mappings into ghost buffers.  Macros are used here because these
 * mappings are used in host code as well as device code.  In C-notation, the
 * mappings defined below are used for bounds:
 *
 * GHOSTX[NodeBlock::sizeZ][NodeBlock::sizeY][3]
 * GHOSTY[NodeBlock::sizeZ][3][NodeBlock::sizeX]
 *
 * The layout is chosen such that a warp access is coalesced
 * */
#define GHOSTMAPX(ix,iy,iz) ((iy) + NodeBlock::sizeY * ((ix) + 3 * (iz)))
#define GHOSTMAPY(ix,iy,iz) ((ix) + NodeBlock::sizeX * ((iy) + 3 * (iz)))

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

// material dictionary
class MaterialDictionary
{
    public:
        static Real rho1;
        static Real rho2;
        static Real gamma1;
        static Real gamma2;
        static Real pc1;
        static Real pc2;
};

// Simulation framework
class Simulation
{
    protected:
        virtual void _setup() = 0;

    public:
        virtual void run() = 0;
        virtual ~Simulation() { }
};


// Simulation Tools
class SimTools
{
    public:
        static double EPSILON;

        static Real heaviside(const Real phi)
        {
            return (phi>0? 0:1);
        }

        static Real heaviside_smooth(const Real phi)
        {
            const Real alpha = M_PI*std::min(1., std::max(0., 0.5*(phi+EPSILON)/EPSILON));
            return 0.5+0.5*std::cos(alpha);
        }

        static void getPostShockRatio(const Real pre_shock[3], const Real mach, const Real gamma, const Real pc, Real postShock[3])
        {
            const double Mpost = std::sqrt( (std::pow(mach,(Real)2.)*(gamma-1.)+2.) / (2.*gamma*std::pow(mach,(Real)2.)-(gamma-1.)) );
            postShock[0] = (gamma+1.)*std::pow(mach,(Real)2.)/( (gamma-1.)*std::pow(mach,(Real)2.)+2.)*pre_shock[0] ;
            postShock[2] = 1./(gamma+1.) * ( 2.*gamma*std::pow(mach,(Real)2.)-(gamma-1.))*pre_shock[2];
            const double preShockU = mach*std::sqrt(gamma*(pc+pre_shock[2])/pre_shock[0]);
            const double postShockU = Mpost*std::sqrt(gamma*(pc+postShock[2])/postShock[0]);
            postShock[1] = preShockU - postShockU;
        }

        template <typename TGrid>
            static void symmetry_check(TGrid& grid, const int verbosity, const Real tol=1.0e-6, uint_t dims[3]=NULL)
            {
                // check for symmetry in minor direction -> dims = {principal, minor, dummy}
                if (dims == NULL)
                    for (uint_t i = 0; i < 3; ++i)
                        dims[i] = i;

                int idx[3];
                int sym[3];
                Real Linf_global = 0;
                const uint_t gridDim[3] = {TGrid::sizeX, TGrid::sizeY, TGrid::sizeZ};
                typedef typename TGrid::PRIM var;
                for (idx[dims[2]]=0; idx[dims[2]]<gridDim[dims[2]]; ++idx[dims[2]])
                    for (idx[dims[1]]=0; idx[dims[1]]<gridDim[dims[1]]/2; ++idx[dims[1]])
                        for (idx[dims[0]]=0; idx[dims[0]]<gridDim[dims[0]]; ++idx[dims[0]])
                        {
                            sym[dims[0]] = idx[dims[0]];
                            sym[dims[1]] = gridDim[dims[1]]-1 - idx[dims[1]];
                            sym[dims[2]] = idx[dims[2]];
                            if (verbosity==3) printf("Symmetry Check: (%d,%d,%d)<->(%d,%d,%d) => ",idx[0],idx[1],idx[2],sym[0],sym[1],sym[2]);
                            Real d0 = std::abs( grid(idx[0],idx[1],idx[2],var::R) - grid(sym[0],sym[1],sym[2],var::R) );
                            Real d1 = std::abs( grid(idx[0],idx[1],idx[2],var::U) - grid(sym[0],sym[1],sym[2],var::U) );
                            Real d2 = std::abs( grid(idx[0],idx[1],idx[2],var::V) - grid(sym[0],sym[1],sym[2],var::V) );
                            Real d3 = std::abs( grid(idx[0],idx[1],idx[2],var::W) - grid(sym[0],sym[1],sym[2],var::W) );
                            Real d4 = std::abs( grid(idx[0],idx[1],idx[2],var::E) - grid(sym[0],sym[1],sym[2],var::E) );
                            Real d5 = std::abs( grid(idx[0],idx[1],idx[2],var::G) - grid(sym[0],sym[1],sym[2],var::G) );
                            Real d6 = std::abs( grid(idx[0],idx[1],idx[2],var::P) - grid(sym[0],sym[1],sym[2],var::P) );
                            assert(d0 < tol);
                            assert(d1 < tol);
                            assert(d2 < tol);
                            assert(d3 < tol);
                            assert(d4 < tol);
                            assert(d5 < tol);
                            assert(d6 < tol);
                            Real Linf = std::max(d0,std::max(d1,std::max(d2,std::max(d3,std::max(d4,std::max(d5,d6))))));
                            Linf_global = std::max(Linf, Linf_global);
                            if (verbosity==3) printf("Linf = %f\n", Linf);
                        }
                printf("Global Linf = %f\n", Linf_global);
            }
};


///////////////////////////////////////////////////////////////////////////////
// OUTPUT STREAMER
///////////////////////////////////////////////////////////////////////////////
struct StreamerGridPointIterative //dummy
{
    static const int channels = 7;

    template<int channel>
    static inline Real operate(const FluidElement& input) { abort(); return 0; }


    const char * name() { return "StreamerGridPointIterative" ; }
};

template<> inline Real StreamerGridPointIterative::operate<0>(const FluidElement& e) { return e.rho; }
template<> inline Real StreamerGridPointIterative::operate<1>(const FluidElement& e) { return e.u/e.rho; }
template<> inline Real StreamerGridPointIterative::operate<2>(const FluidElement& e) { return e.v/e.rho; }
template<> inline Real StreamerGridPointIterative::operate<3>(const FluidElement& e) { return e.w/e.rho; }
template<> inline Real StreamerGridPointIterative::operate<4>(const FluidElement& e) { return (e.energy-static_cast<Real>(0.5)*(e.u*e.u+e.v*e.v+e.w*e.w)/e.rho - e.P)/e.G; }
template<> inline Real StreamerGridPointIterative::operate<5>(const FluidElement& e) { return e.G; }
template<> inline Real StreamerGridPointIterative::operate<6>(const FluidElement& e) { return e.P; }


struct myTensorialStreamer
{
    static const int NCHANNELS = 9;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    myTensorialStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
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

    static const char * getAttributeName() { return "Tensor"; }
};


struct myScalarStreamer
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    myScalarStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = r[idx];
    }

    static const char * getAttributeName() { return "Scalar"; }
};


struct myVelocityStreamer
{
    static const int NCHANNELS = 3;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    myVelocityStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        const Real inv_r = 1.0 / r[idx];
        out[0] = inv_r * u[idx];
        out[1] = inv_r * v[idx];
        out[2] = inv_r * w[idx];
    }

    static const char * getAttributeName() { return "Vector"; }
};


struct myPressureStreamer
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    myPressureStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = (e[idx]-0.5f*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx])/r[idx] - P[idx])/G[idx];
    }

    static const char * getAttributeName() { return "Scalar"; }
};


struct myRhoStreamer
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    myRhoStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = r[idx];
    }

    static const char * getAttributeName() { return "Scalar"; }
};


struct myEnergyStreamer
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    myEnergyStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = e[idx];
    }

    static const char * getAttributeName() { return "Scalar"; }
};


struct myGammaStreamer
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    myGammaStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = G[idx];
    }

    static const char * getAttributeName() { return "Scalar"; }
};


struct myPiStreamer
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef const Real * const const_ptr;
    const_ptr r, u, v, w, e, G, P;

    myPiStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = P[idx];
    }

    static const char * getAttributeName() { return "Scalar"; }
};


struct mySaveStreamer
{
    static const int NCHANNELS = NodeBlock::NVAR;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef Real * const flow_quantity;
    flow_quantity r, u, v, w, e, G, P;

    mySaveStreamer(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = r[idx];
        out[1] = u[idx];
        out[2] = v[idx];
        out[3] = w[idx];
        out[4] = e[idx];
        out[5] = G[idx];
        out[6] = P[idx];
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz)
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        r[idx] = input[0];
        u[idx] = input[1];
        v[idx] = input[2];
        w[idx] = input[3];
        e[idx] = input[4];
        G[idx] = input[5];
        P[idx] = input[6];
    }

    static const char * getAttributeName() { return "Save_Data"; }
};

// Due to 2GB write limitation inside ROMIO (part of MPI lib), save each
// conservative variable in its own file. This is not needed if total save data
// is less than 2GB
// http://lists.hdfgroup.org/pipermail/hdf-forum_lists.hdfgroup.org/2013-January/006447.html
struct mySaveStreamer_part1
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef Real * const flow_quantity;
    flow_quantity r, u, v, w, e, G, P;

    mySaveStreamer_part1(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = r[idx];
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz)
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        r[idx] = input[0];
    }

    static const char * getAttributeName() { return "Save_Data_Part1"; }
};

struct mySaveStreamer_part2
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef Real * const flow_quantity;
    flow_quantity r, u, v, w, e, G, P;

    mySaveStreamer_part2(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = u[idx];
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz)
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        u[idx] = input[0];
    }

    static const char * getAttributeName() { return "Save_Data_Part2"; }
};

struct mySaveStreamer_part3
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef Real * const flow_quantity;
    flow_quantity r, u, v, w, e, G, P;

    mySaveStreamer_part3(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = v[idx];
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz)
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        v[idx] = input[0];
    }

    static const char * getAttributeName() { return "Save_Data_Part3"; }
};

struct mySaveStreamer_part4
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef Real * const flow_quantity;
    flow_quantity r, u, v, w, e, G, P;

    mySaveStreamer_part4(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = w[idx];
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz)
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        w[idx] = input[0];
    }

    static const char * getAttributeName() { return "Save_Data_Part4"; }
};

struct mySaveStreamer_part5
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef Real * const flow_quantity;
    flow_quantity r, u, v, w, e, G, P;

    mySaveStreamer_part5(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = e[idx];
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz)
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        e[idx] = input[0];
    }

    static const char * getAttributeName() { return "Save_Data_Part5"; }
};

struct mySaveStreamer_part6
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef Real * const flow_quantity;
    flow_quantity r, u, v, w, e, G, P;

    mySaveStreamer_part6(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = G[idx];
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz)
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        G[idx] = input[0];
    }

    static const char * getAttributeName() { return "Save_Data_Part6"; }
};

struct mySaveStreamer_part7
{
    static const int NCHANNELS = 1;
    static const int NX = NodeBlock::sizeX;
    static const int NY = NodeBlock::sizeY;
    static const int NZ = NodeBlock::sizeZ;

    typedef Real * const flow_quantity;
    flow_quantity r, u, v, w, e, G, P;

    mySaveStreamer_part7(const std::vector<Real *>& ptr) : r(ptr[0]), u(ptr[1]), v(ptr[2]), w(ptr[3]), e(ptr[4]), G(ptr[5]), P(ptr[6]) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        out[0] = P[idx];
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz)
    {
        const int idx = ID3(ix,iy,iz,NX,NY);
        assert(idx < NX * NY * NZ);
        P[idx] = input[0];
    }

    static const char * getAttributeName() { return "Save_Data_Part7"; }
};

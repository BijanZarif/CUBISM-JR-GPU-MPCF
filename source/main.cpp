/* *
 * main.cpp
 *
 * Created by Fabian Wermelinger on 6/24/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include <fstream>
#include <vector>
#include <cmath>
using namespace std;

#include "ArgumentParser.h"
#include "GridMPI.h"
#include "GPUlab.h"
#include "Types.h"
#include "Timer.h"
#include "HDF5Dumper_MPI.h"
/* #include "SerializerIO_WaveletCompression_MPI_Simple.h" */

#include "MaxSpeedOfSound_CUDA.h"
#include "MaxSpeedOfSound.h"
#include "Convection_CUDA.h"
#include "Update_CUDA.h"
#include "BoundaryConditions.h"


// Helper
template <typename Ksos>
static double _maxSOS(const RealPtrVec_t& src, float& sos)
{
    Ksos kernel;
    Timer tsos;
    tsos.start();
    sos = kernel.compute(src);
    return tsos.stop();
}


template <typename Ksos>
static double _maxSOS(GPUlab * const myGPU, float& sos)
{
    return myGPU->template max_sos<Ksos>(sos);
}


static void _icCONST(GridMPI& grid, const Real val = 0)
{
    typedef GridMPI::PRIM var;
#pragma omp paralell for
    for (int iz = 0; iz < GridMPI::sizeZ; ++iz)
        for (int iy = 0; iy < GridMPI::sizeY; ++iy)
            for (int ix = 0; ix < GridMPI::sizeX; ++ix)
            {
                grid(ix, iy, iz, var::R) = val;
                grid(ix, iy, iz, var::U) = val;
                grid(ix, iy, iz, var::V) = val;
                grid(ix, iy, iz, var::W) = val;
                grid(ix, iy, iz, var::E) = val;
                grid(ix, iy, iz, var::G) = val;
                grid(ix, iy, iz, var::P) = val;
            }

}


static void _ic123(GridMPI& grid)
{
    // 1 2 3 4 5 6 7 8 9 .........
    typedef GridMPI::PRIM var;
    unsigned int cnt = 0;
#pragma omp paralell for
    for (int iz = 0; iz < GridMPI::sizeZ; ++iz)
        for (int iy = 0; iy < GridMPI::sizeY; ++iy)
            for (int ix = 0; ix < GridMPI::sizeX; ++ix)
            {
                grid(ix, iy, iz, var::R) = cnt;
                grid(ix, iy, iz, var::U) = cnt;
                grid(ix, iy, iz, var::V) = cnt;
                grid(ix, iy, iz, var::W) = cnt;
                grid(ix, iy, iz, var::E) = cnt;
                grid(ix, iy, iz, var::G) = cnt;
                grid(ix, iy, iz, var::P) = cnt++;
            }

}


static void _icSOD(GridMPI& grid, ArgumentParser& parser, const unsigned int dims[3])
{
    ///////////////////////////////////////////////////////////////////////////
    // SOD
    ///////////////////////////////////////////////////////////////////////////
    const double x0   = parser("-x0").asDouble(0.5);
    const double rho1 = parser("-rho1").asDouble(1.0);
    const double rho2 = parser("-rho2").asDouble(0.125);
    const double u1   = parser("-u1").asDouble(0.0);
    const double u2   = parser("-u2").asDouble(0.0);
    const double p1   = parser("-p1").asDouble(1.0);
    const double p2   = parser("-p2").asDouble(0.1);
    const double g1   = parser("-g1").asDouble(1.4);
    const double g2   = parser("-g2").asDouble(1.4);
    const double pc1  = parser("-pc1").asDouble(0.0);
    const double pc2  = parser("-pc2").asDouble(0.0);

    typedef GridMPI::PRIM var;
    const var momentum[3] = {var::U, var::V, var::W};
#pragma omp paralell for
    for (int iz = 0; iz < GridMPI::sizeZ; ++iz)
        for (int iy = 0; iy < GridMPI::sizeY; ++iy)
            for (int ix = 0; ix < GridMPI::sizeX; ++ix)
            {
                double pos[3];
                grid.get_pos(ix, iy, iz, pos);

                // set up along x
                bool x = pos[dims[0]] < x0;
                /* bool x = pos[dims[0]] > x0; */

                const double r = x * rho1 + !x * rho2;
                const double p = x * p1   + !x * p2;
                const double u = x * rho1*u1 + !x * rho2*u2;
                const double G = x * 1./(g1 - 1.) + !x * 1./(g2 - 1.);
                const double P = x * pc1*g1/(g1 - 1.) + !x * pc2*g2/(g2 - 1.);
                assert(r > 0);
                assert(p > 0);
                assert(G > 0);
                assert(P >= 0);

                grid(ix, iy, iz, var::R) = r;
                grid(ix, iy, iz, momentum[dims[0]]) = u;
                grid(ix, iy, iz, momentum[dims[1]]) = 0;
                grid(ix, iy, iz, momentum[dims[2]]) = 0;
                grid(ix, iy, iz, var::E) = G*p + P + 0.5*u*u/r;
                grid(ix, iy, iz, var::G) = G;
                grid(ix, iy, iz, var::P) = P;
            }
}


Real EPSILON = 1.0;

static Real _heaviside(const Real phi)
{
    return (phi>0? 0:1);
}

static Real _heaviside_smooth(const Real phi)
{
    const Real alpha = M_PI*min(1., max(0., 0.5*(phi+EPSILON)/EPSILON));
    return 0.5+0.5*cos(alpha);
}

static void _getPostShockRatio(const Real pre_shock[3], const Real mach, const Real gamma, const Real pc, Real postShock[3])
{
    const double Mpost = sqrt( (pow(mach,(Real)2.)*(gamma-1.)+2.) / (2.*gamma*pow(mach,(Real)2.)-(gamma-1.)) );
    postShock[0] = (gamma+1.)*pow(mach,(Real)2.)/( (gamma-1.)*pow(mach,(Real)2.)+2.)*pre_shock[0] ;
    postShock[2] = 1./(gamma+1.) * ( 2.*gamma*pow(mach,(Real)2.)-(gamma-1.))*pre_shock[2];
    const double preShockU = mach*sqrt(gamma*(pc+pre_shock[2])/pre_shock[0]);
    const double postShockU = Mpost*sqrt(gamma*(pc+postShock[2])/postShock[0]);
    postShock[1] = preShockU - postShockU;
}

static void _ic2DSB(GridMPI& grid, ArgumentParser& parser, const unsigned int dims[3])
{
    ///////////////////////////////////////////////////////////////////////////
    // 2D Shock Bubble
    ///////////////////////////////////////////////////////////////////////////
    const double x0   = parser("-x0").asDouble(0.1);
    const double mach = parser("-mach").asDouble(1.22);
    const double rho2 = parser("-rho").asDouble(1);
    const double rhob = parser("-rhobubble").asDouble(0.138);
    const double u2   = parser("-u").asDouble(0.0);
    const double p2   = parser("-p").asDouble(1);
    const double pbub = parser("-pbubble").asDouble(1);
    const double g1   = parser("-g1").asDouble(1.4);
    const double g2   = parser("-g2").asDouble(1.67);
    const double pc1  = parser("-pc1").asDouble(0.0);
    const double pc2  = parser("-pc2").asDouble(0.0);

    const Real pre_shock[3] = {rho2, u2, p2};
    Real post_shock[3];
    _getPostShockRatio(pre_shock, mach, g1, pc1, post_shock);
    const double rho1 = post_shock[0];
    const double u1   = post_shock[1];
    const double p1   = post_shock[2];
    printf("Post Shock Velocity = %f\n", u1);

    EPSILON = (Real)(parser("-mollfactor").asInt(1))*sqrt(3.)*grid.getH();

    const double G1 = g1-1;
    const double G2 = g2-1;
    const double F1 = g1*pc1;
    const double F2 = g2*pc2;

    vector<Real> Rb;       // radius of bubble
    vector<vector<Real> > posb;  // position of bubble

    // read bubbles from file
    char fname[256];
    if (dims[0]==0)
        sprintf(fname, dims[1]==1 ? "xybubbles.dat" : "xzbubbles.dat");
    else if (dims[0]==1)
        sprintf(fname, dims[1]==0 ? "yxbubbles.dat" : "yzbubbles.dat");
    else if (dims[0]==2)
        sprintf(fname, dims[1]==0 ? "zxbubbles.dat" : "zybubbles.dat");
    ifstream bubbleFile(fname);
    if (!bubbleFile.good())
    {
        fprintf(stderr, "Can not load bubble file './%s'. ABORT\n", fname);
        exit(1);
    }
    unsigned int nb;
    bubbleFile >> nb;
    Rb.resize(nb);
    posb.resize(nb);
    for (int i = 0; i < nb; ++i)
    {
        if (!bubbleFile.good())
        {
            fprintf(stderr, "Error reading './%s'. ABORT\n", fname);
            exit(1);
        }
        posb[i].resize(3);
        bubbleFile >> posb[i][0];
        bubbleFile >> posb[i][1];
        bubbleFile >> posb[i][2];
        bubbleFile >> Rb[i];
    }
    bubbleFile.close();

    // init
    typedef GridMPI::PRIM var;
    const var momentum[3] = {var::U, var::V, var::W};
#pragma omp paralell for
    for (int iz = 0; iz < GridMPI::sizeZ; ++iz)
        for (int iy = 0; iy < GridMPI::sizeY; ++iy)
            for (int ix = 0; ix < GridMPI::sizeX; ++ix)
            {
                double p[3];
                grid.get_pos(ix, iy, iz, p);

                // determine shock region
                const Real shock = _heaviside(p[dims[0]] - x0);

                // process all bubbles
                Real bubble;
                for (int i = 0; i < nb; ++i)
                {
                    const Real r = sqrt( pow(p[dims[0]] - posb[i][dims[0]], 2) + pow(p[dims[1]] - posb[i][dims[1]], 2) );
                    bubble = _heaviside_smooth(r - Rb[i]);
                    if (bubble > 0)
                        break;
                }

                grid(ix, iy, iz, var::R) = shock*rho1 + (1 - shock)*(rhob*bubble + rho2*(1 - bubble));

                // even if specified, bubbles have same IC velocity as
                // bulk flow.
                grid(ix, iy, iz, momentum[dims[0]]) = (shock*u1 + (1-shock)*u2) * grid(ix, iy, iz, var::R);
                grid(ix, iy, iz, momentum[dims[1]]) = 0;
                grid(ix, iy, iz, momentum[dims[2]]) = 0;

                // phase mix
                grid(ix, iy, iz, var::G) = 1./G1*(1 - bubble) + 1./G2*bubble;
                grid(ix, iy, iz, var::P) = F1/G1*(1 - bubble) + F2/G2*bubble;

                // energy
                const Real pressure  = shock*p1 + (1-shock)*(pbub*bubble + p2*(1 - bubble));
                const Real ke = 0.5*(pow(grid(ix, iy, iz, var::U),2)+pow(grid(ix, iy, iz, var::V),2)+pow(grid(ix, iy, iz, var::W),2))/grid(ix, iy, iz, var::R);
                grid(ix, iy, iz, var::E) = pressure*grid(ix, iy, iz, var::G) + grid(ix, iy, iz, var::P) + ke;
            }
}


void _symmetry_check(GridMPI& grid, const int verbose, const Real tol=1.0e-6, unsigned int dims[3]=NULL)
{
    // check for symmetry in minor direction
    if (dims == NULL)
        for (unsigned int i = 0; i < 3; ++i)
            dims[i] = i;

    int idx[3];
    int sym[3];
    Real Linf_global = 0;
    const unsigned int gridDim[3] = {GridMPI::sizeX, GridMPI::sizeY, GridMPI::sizeZ};
    typedef GridMPI::PRIM var;
    for (idx[dims[2]]=0; idx[dims[2]]<gridDim[dims[2]]; ++idx[dims[2]])
        for (idx[dims[1]]=0; idx[dims[1]]<gridDim[dims[1]]/2; ++idx[dims[1]])
            for (idx[dims[0]]=0; idx[dims[0]]<gridDim[dims[0]]; ++idx[dims[0]])
            {
                sym[dims[0]] = idx[dims[0]];
                sym[dims[1]] = gridDim[dims[1]]-1 - idx[dims[1]];
                sym[dims[2]] = idx[dims[2]];
                if (verbose) printf("Symmetry Check: (%d,%d,%d)<->(%d,%d,%d) => ",idx[0],idx[1],idx[2],sym[0],sym[1],sym[2]);
                Real d0 = abs( grid(idx[0],idx[1],idx[2],var::R) - grid(sym[0],sym[1],sym[2],var::R) );
                Real d1 = abs( grid(idx[0],idx[1],idx[2],var::U) - grid(sym[0],sym[1],sym[2],var::U) );
                Real d2 = abs( grid(idx[0],idx[1],idx[2],var::V) - grid(sym[0],sym[1],sym[2],var::V) );
                Real d3 = abs( grid(idx[0],idx[1],idx[2],var::W) - grid(sym[0],sym[1],sym[2],var::W) );
                Real d4 = abs( grid(idx[0],idx[1],idx[2],var::E) - grid(sym[0],sym[1],sym[2],var::E) );
                Real d5 = abs( grid(idx[0],idx[1],idx[2],var::G) - grid(sym[0],sym[1],sym[2],var::G) );
                Real d6 = abs( grid(idx[0],idx[1],idx[2],var::P) - grid(sym[0],sym[1],sym[2],var::P) );
                assert(d0 < tol);
                assert(d1 < tol);
                assert(d2 < tol);
                assert(d3 < tol);
                assert(d4 < tol);
                assert(d5 < tol);
                assert(d6 < tol);
                /* if (verbose) printf("abs-diff: (%f, %f, %f, %f, %f, %f, %f)\n",d0,d1,d2,d3,d4,d5,d6); */
                Real Linf = max(d0,max(d1,max(d2,max(d3,max(d4,max(d5,d6))))));
                Linf_global = max(Linf, Linf_global);
                if (verbose) printf("Linf = %f\n", Linf);

                /* assert(grid(idx[0],idx[1],idx[2],var::R) == grid(sym[0],sym[1],sym[2],var::R)); */
                /* assert(grid(idx[0],idx[1],idx[2],var::U) == grid(sym[0],sym[1],sym[2],var::U)); */
                /* assert(grid(idx[0],idx[1],idx[2],var::V) == grid(sym[0],sym[1],sym[2],var::V)); */
                /* assert(grid(idx[0],idx[1],idx[2],var::W) == grid(sym[0],sym[1],sym[2],var::W)); */
                /* assert(grid(idx[0],idx[1],idx[2],var::E) == grid(sym[0],sym[1],sym[2],var::E)); */
                /* assert(grid(idx[0],idx[1],idx[2],var::G) == grid(sym[0],sym[1],sym[2],var::G)); */
                /* assert(grid(idx[0],idx[1],idx[2],var::P) == grid(sym[0],sym[1],sym[2],var::P)); */
            }
    printf("Global Linf = %f\n", Linf_global);
}


template <typename TGPU>
static inline double _LSRKstep(const Real a, const Real b, const Real dtinvh, TGPU& gpu)
{
    return gpu.template process_all<Convection_CUDA, Update_CUDA>(a, b, dtinvh);
}


// implement boundary conditions for SOD
class GPUlabSOD : public GPUlab
{
    protected:
        void _apply_bc(const double t = 0)
        {
            /* BoundaryConditions<GridMPI> bc(grid.pdata(), current_iz, current_length); */
            BoundaryConditions<GridMPI> bc(grid.pdata()); // call this constructor if all halos are fetched at one time
            if (myFeature[0] == SKIN) bc.template applyBC_absorbing<0,0,ghostmap::X>(halox.left);
            if (myFeature[1] == SKIN) bc.template applyBC_absorbing<0,1,ghostmap::X>(halox.right);
            if (myFeature[2] == SKIN) bc.template applyBC_absorbing<1,0,ghostmap::Y>(haloy.left);
            if (myFeature[3] == SKIN) bc.template applyBC_absorbing<1,1,ghostmap::Y>(haloy.right);
            if (myFeature[4] == SKIN) bc.template applyBC_absorbing<2,0,ghostmap::Z>(haloz.left);
            if (myFeature[5] == SKIN) bc.template applyBC_absorbing<2,1,ghostmap::Z>(haloz.right);
        }

    public:
        GPUlabSOD(GridMPI& grid, const unsigned int nslices, const int verb) : GPUlab(grid, nslices, verb) { }
};


class GPUlabSB : public GPUlab
{
    protected:
        void _apply_bc(const double t = 0)
        {
            /* BoundaryConditions<GridMPI> bc(grid.pdata(), current_iz, current_length); */
            BoundaryConditions<GridMPI> bc(grid.pdata()); // call this constructor if all halos are fetched at one time
            // xy / zy
            /* if (myFeature[0] == SKIN) bc.template applyBC_absorbing <0,0,ghostmap::X>(halox.left); */
            /* if (myFeature[1] == SKIN) bc.template applyBC_absorbing <0,1,ghostmap::X>(halox.right); */
            /* if (myFeature[2] == SKIN) bc.template applyBC_reflecting<1,0,ghostmap::Y>(haloy.left); */
            /* if (myFeature[3] == SKIN) bc.template applyBC_reflecting<1,1,ghostmap::Y>(haloy.right); */
            /* if (myFeature[4] == SKIN) bc.template applyBC_absorbing <2,0,ghostmap::Z>(haloz.left); */
            /* if (myFeature[5] == SKIN) bc.template applyBC_absorbing <2,1,ghostmap::Z>(haloz.right); */

            // yx / zx
            /* if (myFeature[0] == SKIN) bc.template applyBC_reflecting<0,0,ghostmap::X>(halox.left); */
            /* if (myFeature[1] == SKIN) bc.template applyBC_reflecting<0,1,ghostmap::X>(halox.right); */
            /* if (myFeature[2] == SKIN) bc.template applyBC_absorbing <1,0,ghostmap::Y>(haloy.left); */
            /* if (myFeature[3] == SKIN) bc.template applyBC_absorbing <1,1,ghostmap::Y>(haloy.right); */
            /* if (myFeature[4] == SKIN) bc.template applyBC_absorbing <2,0,ghostmap::Z>(haloz.left); */
            /* if (myFeature[5] == SKIN) bc.template applyBC_absorbing <2,1,ghostmap::Z>(haloz.right); */

            // yz / xz
            if (myFeature[0] == SKIN) bc.template applyBC_absorbing <0,0,ghostmap::X>(halox.left);
            if (myFeature[1] == SKIN) bc.template applyBC_absorbing <0,1,ghostmap::X>(halox.right);
            if (myFeature[2] == SKIN) bc.template applyBC_absorbing <1,0,ghostmap::Y>(haloy.left);
            if (myFeature[3] == SKIN) bc.template applyBC_absorbing <1,1,ghostmap::Y>(haloy.right);
            if (myFeature[4] == SKIN) bc.template applyBC_reflecting<2,0,ghostmap::Z>(haloz.left);
            if (myFeature[5] == SKIN) bc.template applyBC_reflecting<2,1,ghostmap::Z>(haloz.right);
        }

    public:
        GPUlabSB(GridMPI& grid, const unsigned int nslices, const int verb) : GPUlab(grid, nslices, verb) { }
};

/* typedef GPUlabSOD Lab; */
typedef GPUlabSB Lab;


const char *_make_fname(char *fname, const char *base, const int fcount)
{
    sprintf(fname, "%s_%04d", base, fcount);
    return fname;
}



int main(int argc, const char *argv[])
{
    MPI_Init(&argc, const_cast<char***>(&argv));

    /* int provided; */
    /* MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); */

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const bool isroot = world_rank == 0;

    ArgumentParser parser(argc, argv);

    const int verbose = parser("-verbose").asInt(0);

    ///////////////////////////////////////////////////////////////////////////
    // Setup MPI grid
    ///////////////////////////////////////////////////////////////////////////
    const int npex = parser("-npex").asInt(1);
    const int npey = parser("-npey").asInt(1);
    const int npez = parser("-npez").asInt(1);
    GridMPI mygrid(npex, npey, npez);

    ///////////////////////////////////////////////////////////////////////////
    // Setup Initial Condition
    ///////////////////////////////////////////////////////////////////////////
    unsigned int dims[3] = {0,2,1}; // permutation of directions for 1D/2D stuff {principal, minor, dummy}
    /* _icCONST(mygrid, world_rank+1); */
    /* _ic123(mygrid); */
    /* _icSOD(mygrid, parser, dims); */
    _ic2DSB(mygrid, parser, dims);

    const Real tol = 1.0e-6;
    _symmetry_check(mygrid, verbose, tol, dims);

    unsigned int fcount = 0;
    char fname[256];
    DumpHDF5_MPI<GridMPI, myTensorialStreamer>(mygrid, 0, _make_fname(fname, "data", fcount++));

    ///////////////////////////////////////////////////////////////////////////
    // Init GPU
    ///////////////////////////////////////////////////////////////////////////
    const size_t nslices = 128;
    Lab myGPU(mygrid, nslices, verbose);

    ///////////////////////////////////////////////////////////////////////////
    // Run Solver
    ///////////////////////////////////////////////////////////////////////////
    parser.set_strict_mode();
    const double tend = parser("-tend").asDouble();
    const double cfl  = parser("-cfl").asDouble();
    parser.unset_strict_mode();
    const unsigned int nsteps = parser("-nsteps").asInt(0);

    const double h = mygrid.getH();
    float sos;
    double t = 0, dt, tlast = 0;
    unsigned int step = 0;

    while (t < tend)
    {
        // 1.) Compute max SOS -> dt
        const double tsos = _maxSOS<MaxSpeedOfSound_CUDA>(&myGPU, sos);
        /* const double tsos = _maxSOS<MaxSpeedOfSound_CPP>(mygrid.pdata(), sos); */
        assert(sos > 0);
        if (verbose) printf("sos = %f (took %f sec)\n", sos, tsos);

        dt = cfl*h/sos;
        dt = (tend-t) < dt ? (tend-t) : dt;
        MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, mygrid.getCartComm());

        // 2.) Compute RHS and update using LSRK3
        double trk1, trk2, trk3;
        {// stage 1
            myGPU.load_ghosts();
            trk1 = _LSRKstep<Lab>(0, 1./4, dt/h, myGPU);
            if (verbose) printf("RK stage 1 takes %f sec\n", trk1);
        }
        {// stage 2
            myGPU.load_ghosts();
            trk2 = _LSRKstep<Lab>(-17./32, 8./9, dt/h, myGPU);
            if (verbose) printf("RK stage 2 takes %f sec\n", trk2);
        }
        {// stage 3
            myGPU.load_ghosts();
            trk3 = _LSRKstep<Lab>(-32./27, 3./4, dt/h, myGPU);
            if (verbose) printf("RK stage 3 takes %f sec\n", trk3);
        }
        if (verbose) printf("netto step takes %f sec\n", tsos + trk1 + trk2 + trk3);

        t += dt;
        ++step;

        printf("step id is %d, physical time %f (dt = %f)\n", step, t, dt);

        if ((t-tlast)*1000 > 1.0 && (int)(t*1000) % 6 == 0)
        {
            tlast = t;
            /* DumpHDF5_MPI<GridMPI, myTensorialStreamer>(mygrid, step, _make_fname(fname, "data", fcount++)); */
        }
        /* if (step % 10 == 0) DumpHDF5_MPI<GridMPI, myTensorialStreamer>(mygrid, step, _make_fname(fname, "data", fcount++)); */

        if (step == nsteps) break;
    }

    DumpHDF5_MPI<GridMPI, myTensorialStreamer>(mygrid, step, _make_fname(fname, "data", fcount++));

    // good night
    MPI_Finalize();

    return 0;
}

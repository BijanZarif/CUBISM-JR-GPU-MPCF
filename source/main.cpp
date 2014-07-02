/* *
 * main.cpp
 *
 * Created by Fabian Wermelinger on 6/24/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <stdio.h>
#include <assert.h>
#include <omp.h>

#include "ArgumentParser.h"
#include "GridMPI.h"
#include "GPUlab.h"
#include "Types.h"
#include "Timer.h"
#include "HDF5Dumper_MPI.h"
/* #include "SerializerIO_WaveletCompression_MPI_Simple.h" */

#include "MaxSpeedOfSound_CUDA.h"
#include "Convection_CUDA.h"
#include "Update_CUDA.h"
#include "BoundaryConditions.h"


// Helper
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


static void _icSOD(GridMPI& grid, ArgumentParser& parser)
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
#pragma omp paralell for
    for (int iz = 0; iz < GridMPI::sizeZ; ++iz)
        for (int iy = 0; iy < GridMPI::sizeY; ++iy)
            for (int ix = 0; ix < GridMPI::sizeX; ++ix)
            {
                double pos[3];
                grid.get_pos(ix, iy, iz, pos);

                // set up along x
                bool x = pos[2] < x0;

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
                grid(ix, iy, iz, var::W) = u;
                grid(ix, iy, iz, var::V) = 0;
                grid(ix, iy, iz, var::U) = 0;
                grid(ix, iy, iz, var::E) = G*p + P + 0.5*u*u/r;
                grid(ix, iy, iz, var::G) = G;
                grid(ix, iy, iz, var::P) = P;
            }
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
            if (myFeature[0] == SKIN) bc.template applyBC_absorbing<0,0, halomap_x<0,sizeY,3> >(halox.left);
            if (myFeature[1] == SKIN) bc.template applyBC_absorbing<0,1, halomap_x<0,sizeY,3> >(halox.right);
            if (myFeature[2] == SKIN) bc.template applyBC_absorbing<1,0, halomap_y<0,sizeX,3> >(haloy.left);
            if (myFeature[3] == SKIN) bc.template applyBC_absorbing<1,1, halomap_y<0,sizeX,3> >(haloy.right);
            if (myFeature[4] == SKIN) bc.template applyBC_absorbing<2,0, halomap_z<0,sizeX,sizeY> >(haloz.left);
            if (myFeature[5] == SKIN) bc.template applyBC_absorbing<2,1, halomap_z<0,sizeX,sizeY> >(haloz.right);
        }

    public:
        GPUlabSOD(GridMPI& grid, const unsigned int nslices) : GPUlab(grid, nslices) { }
};



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
    /* _icCONST(mygrid, world_rank+1); */
    /* _ic123(mygrid); */
    _icSOD(mygrid, parser);
    /* DumpHDF5_MPI<GridMPI, myTensorialStreamer>(mygrid, 0, "IC"); */

    ///////////////////////////////////////////////////////////////////////////
    // Init GPU
    ///////////////////////////////////////////////////////////////////////////
    const size_t chunk_slices = 64;
    GPUlabSOD myGPU(mygrid, chunk_slices);
    /* GPUlab myGPU(mygrid, chunk_slices); */

    /* myGPU.toggle_verbosity(); */

    typedef GPUlabSOD Lab;
    /* typedef GPUlab Lab; */

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
    double t = 0, dt;
    unsigned int step = 0;

    while (t < tend)
    {
        // 1.) Compute max SOS -> dt
        const double tsos = myGPU.template max_sos<MaxSpeedOfSound_CUDA>(sos);
        assert(sos > 0);
        printf("sos = %f (took %f sec)\n", sos, tsos);

        dt = cfl*h/sos;
        dt = (tend-t) < dt ? (tend-t) : dt;
        MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, mygrid.getCartComm());

        // 2.) Compute RHS and update using LSRK3
        double trk1, trk2, trk3;
        {// stage 1
            myGPU.load_ghosts();
            trk1 = _LSRKstep<Lab>(0, 1./4, dt/h, myGPU);
            printf("RK stage 1 takes %f sec\n", trk1);
        }
        {// stage 2
            myGPU.load_ghosts();
            trk2 = _LSRKstep<Lab>(-17./32, 8./9, dt/h, myGPU);
            printf("RK stage 2 takes %f sec\n", trk2);
        }
        {// stage 3
            myGPU.load_ghosts();
            trk3 = _LSRKstep<Lab>(-32./27, 3./4, dt/h, myGPU);
            printf("RK stage 3 takes %f sec\n", trk3);
        }
        printf("netto step takes %f sec\n", tsos + trk1 + trk2 + trk3);

        t += dt;
        ++step;

        printf("step id is %d, physical time %f (dt = %f)\n", step, t, dt);

        if (step == nsteps)
            break;
    }

    DumpHDF5_MPI<GridMPI, myTensorialStreamer>(mygrid, 0, "final");

    // good night
    MPI_Finalize();

    return 0;
}

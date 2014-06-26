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
#include "GPUProcessing.h"
#include "Types.h"
#include "Timer.h"
#include "HDF5Dumper_MPI.h"
#include "SerializerIO_WaveletCompression_MPI_Simple.h"

#include "MaxSpeedOfSound_CUDA.h"
#include "Convection_CUDA.h"
#include "Update_CUDA.h"


// Helper
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


static void _ic(GridMPI& grid, ArgumentParser& parser)
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
                bool x = pos[0] < x0;

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
                grid(ix, iy, iz, var::U) = u;
                grid(ix, iy, iz, var::V) = 0;
                grid(ix, iy, iz, var::W) = 0;
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
    _ic123(mygrid);
    /* _ic(mygrid, parser); */
    /* DumpHDF5_MPI<GridMPI, myTensorialStreamer>(mygrid, 0, "IC"); */

    ///////////////////////////////////////////////////////////////////////////
    // Run Solver
    ///////////////////////////////////////////////////////////////////////////
    const size_t chunk_slices = 64;
    GPUProcessing<GridMPI> myGPU(mygrid, chunk_slices);
    /* myGPU.toggle_verbosity(); */

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
            trk1 = _LSRKstep<GPUProcessing<GridMPI> >(0      , 1./4, dt/h, myGPU);
            printf("RK stage 1 takes %f sec\n", trk1);
        }
        {// stage 2
            trk2 = _LSRKstep<GPUProcessing<GridMPI> >(-17./32, 8./9, dt/h, myGPU);
            printf("RK stage 2 takes %f sec\n", trk2);
        }
        {// stage 3
            trk3 = _LSRKstep<GPUProcessing<GridMPI> >(-32./27, 3./4, dt/h, myGPU);
            printf("RK stage 3 takes %f sec\n", trk3);
        }
        printf("netto step takes %f sec\n", tsos + trk1 + trk2 + trk3);

        t += dt;
        ++step;

        if (step == nsteps)
            break;
    }


    // good night
    MPI_Finalize();

    return 0;
}

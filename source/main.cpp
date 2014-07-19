/* *
 * main.cpp
 *
 * Created by Fabian Wermelinger on 6/24/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "ArgumentParser.h"
#include "Sim_SteadyStateMPI.h"
#include "Sim_SodMPI.h"
#include "Sim_2DSBIMPI.h"

#include <stdio.h>
#include <assert.h>

// Helper

/* template <typename Ksos> */
/* static double _maxSOS(const RealPtrVec_t& src, float& sos) */
/* { */
/*     Ksos kernel; */
/*     Timer tsos; */
/*     tsos.start(); */
/*     sos = kernel.compute(src); */
/*     return tsos.stop(); */
/* } */


/* static void _ic123(GridMPI& grid, const uint_t dims[3]) */
/* { */
/*     // 1 2 3 4 5 6 7 8 9 ......... */
/*     typedef GridMPI::PRIM var; */
/*     uint_t cnt = 0; */
/*     uint_t gridDim[3] = {GridMPI::sizeX, GridMPI::sizeY, GridMPI::sizeZ}; */
/*     int idx[3]; */
/* #pragma omp paralell for */
/*     for (idx[dims[2]]=0; idx[dims[2]]<gridDim[dims[2]]; ++idx[dims[2]]) */
/*         for (idx[dims[1]]=0; idx[dims[1]]<gridDim[dims[1]]; ++idx[dims[1]]) */
/*             for (idx[dims[0]]=0; idx[dims[0]]<gridDim[dims[0]]; ++idx[dims[0]]) */
/*             { */
/*                 grid(idx[0], idx[1], idx[2], var::R) = cnt; */
/*                 grid(idx[0], idx[1], idx[2], var::U) = cnt; */
/*                 grid(idx[0], idx[1], idx[2], var::V) = cnt; */
/*                 grid(idx[0], idx[1], idx[2], var::W) = cnt; */
/*                 grid(idx[0], idx[1], idx[2], var::E) = cnt; */
/*                 grid(idx[0], idx[1], idx[2], var::G) = cnt; */
/*                 grid(idx[0], idx[1], idx[2], var::P) = cnt++; */
/*             } */

/* } */


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
    Simulation *mysim;
    if (parser("-sim").asString("SteadyStateMPI") == "SteadyStateMPI")
        mysim = new Sim_SteadyStateMPI(argc, argv, isroot);
    else if (parser("-sim").asString("SteadyStateMPI") == "SodMPI")
        mysim = new Sim_SodMPI(argc, argv, isroot);
    else if (parser("-sim").asString("SteadyStateMPI") == "2DSBIMPI")
        mysim = new Sim_2DSBIMPI(argc, argv, isroot);

    // setup & run
    mysim->setup();
    mysim->run();

    // good night
    delete mysim;
    MPI_Finalize();

    return 0;
}

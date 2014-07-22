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
#include "Sim_StaticIC.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
using namespace std;


int main(int argc, const char *argv[])
{
    MPI_Init(&argc, const_cast<char***>(&argv));

    /* int provided; */
    /* MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); */

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const bool isroot = world_rank == 0;

    if (isroot)
    {
        for (int i = 0; i < argc; ++i)
            printf("%s ", argv[i]);
        printf("\n");
    }

    ArgumentParser parser(argc, argv);

    parser.set_strict_mode();
    const string select = parser("-sim").asString("SteadyStateMPI");
    parser.unset_strict_mode();

    Simulation *mysim;
    if (select == "SteadyStateMPI")
        mysim = new Sim_SteadyStateMPI(argc, argv, isroot);
    else if (select == "SodMPI")
        mysim = new Sim_SodMPI(argc, argv, isroot);
    else if (select == "2DSBIMPI")
        mysim = new Sim_2DSBIMPI(argc, argv, isroot);
    else if (select == "StaticIC")
        mysim = new Sim_StaticIC(argc, argv, isroot);
    else
    {
        fprintf(stderr, "Error: Unknown simulation case %s...\n", select.c_str());
        exit(1);
    }

    // setup & run
    mysim->run();

    // good night
    delete mysim;
    MPI_Finalize();

    return 0;
}

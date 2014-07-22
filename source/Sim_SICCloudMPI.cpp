/* *
 * Sim_SICCloudMPI.cpp
 *
 * Created by Fabian Wermelinger on 7/22/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "Sim_SICCloudMPI.h"

Sim_SICCloudMPI::Sim_SICCloudMPI(const int argc, const char ** argv, const int isroot) :
    Sim_SteadyStateMPI(argc, argv, isroot)
{
}

void Sim_SICCloudMPI::_allocGPU()
{
    if (isroot) printf("Allocating GPUlabSICCloud...\n");
    myGPU = new GPUlabSICCloud(*mygrid, nslices, verbosity);
}

void Sim_SICCloudMPI::_ic()
{
}

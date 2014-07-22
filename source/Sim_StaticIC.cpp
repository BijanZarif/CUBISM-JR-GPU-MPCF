/* *
 * Sim_StaticIC.cpp
 *
 * Created by Fabian Wermelinger on 7/22/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "Sim_StaticIC.h"
#include "GridMPI.h"
#include <omp.h>
#include <stdlib.h>

Sim_StaticIC::Sim_StaticIC(const int argc, const char **argv, const int isroot) :
    Sim_SteadyStateMPI(argc, argv, isroot)
{
    parser.set_strict_mode();
    const string conf = parser("-config").asString();
    parser.unset_strict_mode();
    if (conf == "xy")
    {
        dims[0] = Coord::X;
        dims[1] = Coord::Y;
        dims[2] = Coord::Z;
    }
    else if (conf == "xz")
    {
        dims[0] = Coord::X;
        dims[2] = Coord::Y;
        dims[1] = Coord::Z;
    }
    else if (conf == "yx")
    {
        dims[1] = Coord::X;
        dims[0] = Coord::Y;
        dims[2] = Coord::Z;
    }
    else if (conf == "yz")
    {
        dims[2] = Coord::X;
        dims[0] = Coord::Y;
        dims[1] = Coord::Z;
    }
    else if (conf == "zx")
    {
        dims[1] = Coord::X;
        dims[2] = Coord::Y;
        dims[0] = Coord::Z;
    }
    else if (conf == "zy")
    {
        dims[2] = Coord::X;
        dims[1] = Coord::Y;
        dims[0] = Coord::Z;
    }
    else
    {
        fprintf(stderr, "ERROR: -config %s is not possible...\n", conf.c_str());
        exit(1);
    }
}


void Sim_StaticIC::_setup()
{
    npex = parser("-npex").asInt(1);
    npey = parser("-npey").asInt(1);
    npez = parser("-npez").asInt(1);
    filename = parser("-fname").asString("staticIC");

    mygrid = new GridMPI(npex, npey, npez);
    assert(mygrid != NULL);
    _ic();
    _dump(filename);
}


void Sim_StaticIC::_ic()
{
    typedef GridMPI::PRIM var;
    uint_t gridDim[3] = {GridMPI::sizeX, GridMPI::sizeY, GridMPI::sizeZ};
    int idx[3];
    uint_t cnt = 0;
    GridMPI& grid = *mygrid;

#pragma omp paralell for
    for (idx[dims[2]]=0; idx[dims[2]]<gridDim[dims[2]]; ++idx[dims[2]])
        for (idx[dims[1]]=0; idx[dims[1]]<gridDim[dims[1]]; ++idx[dims[1]])
            for (idx[dims[0]]=0; idx[dims[0]]<gridDim[dims[0]]; ++idx[dims[0]])
            {
                grid(idx[0], idx[1], idx[2], var::R) = cnt;
                grid(idx[0], idx[1], idx[2], var::U) = cnt;
                grid(idx[0], idx[1], idx[2], var::V) = cnt;
                grid(idx[0], idx[1], idx[2], var::W) = cnt;
                grid(idx[0], idx[1], idx[2], var::E) = cnt;
                grid(idx[0], idx[1], idx[2], var::G) = cnt;
                grid(idx[0], idx[1], idx[2], var::P) = cnt++;
            }
}

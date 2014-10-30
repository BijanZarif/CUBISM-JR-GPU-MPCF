/* *
 * Sim_SteadyStateMPI.h
 *
 * Created by Fabian Wermelinger on 7/18/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include <string>

#include "ArgumentParser.h"
#include "Types.h"
#include "Profiler.h"
#include "GridMPI.h"
#include "GPUlabMPI.h"
#include "LSRK3_IntegratorMPI.h"
#include "BoundaryConditions.h"
#include "SubGridWrapper.h"
#include "SerializerIO_WaveletCompression_MPI_Simple.h"


class Sim_SteadyStateMPI : public Simulation
{
protected:

    const bool isroot;

    // simulation parameter
    double tend, tnextdump, dumpinterval, CFL, maxextent;
    uint_t nsteps, nslices, saveperiod, fcount;
    int verbosity;
    bool restart, dryrun, bIO, bHDF, bVP;
    char fname[256];

    // use artificial subgrid for refined IC setup and wavelet dumps
    SubGridWrapper subblocks;

    // wavelet dumper
    SerializerIO_WaveletCompression_MPI_SimpleBlocking<SubGridWrapper, StreamerGridPointIterative> mywaveletdumper;

    // MPI cartesian grid extent
    uint_t npex, npey, npez;

    // main ingredients
    GridMPI             *mygrid;
    LSRK3_IntegratorMPI *stepper;
    GPUlabMPI           *myGPU;

    // helper
    ArgumentParser parser;
    Profiler& profiler;

    virtual void _setup();
    virtual void _allocGPU();
    virtual void _set_constants();
    virtual void _ic();

    virtual void _dump(const std::string basename = "data");
    virtual void _vp(const std::string basename = "datawavelet");

    virtual void _save();
    virtual bool _restart();

    // release some stuff
    virtual void _take_a_dump(const std::string basename = "data");

public:

    Sim_SteadyStateMPI(const int argc, const char ** argv, const int isroot);
    virtual ~Sim_SteadyStateMPI()
    {
        delete mygrid;
        delete stepper;
        delete myGPU;
    }

    virtual void run();
};


class GPUlabMPISteadyState : public GPUlabMPI
{
protected:
    void _apply_bc(const double t = 0)
    {
        BoundaryConditions<GridMPI> bc(grid.pdata());
        if (myFeature[0] == SKIN) bc.template applyBC_absorbing<0,0,ghostmap::X>(halox.left);
        if (myFeature[1] == SKIN) bc.template applyBC_absorbing<0,1,ghostmap::X>(halox.right);
        if (myFeature[2] == SKIN) bc.template applyBC_absorbing<1,0,ghostmap::Y>(haloy.left);
        if (myFeature[3] == SKIN) bc.template applyBC_absorbing<1,1,ghostmap::Y>(haloy.right);
        if (myFeature[4] == SKIN) bc.template applyBC_absorbing<2,0,ghostmap::Z>(haloz.left);
        if (myFeature[5] == SKIN) bc.template applyBC_absorbing<2,1,ghostmap::Z>(haloz.right);
    }

public:
    GPUlabMPISteadyState(GridMPI& grid, const uint_t nslices, const int verb) : GPUlabMPI(grid, nslices, verb) { }
};

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
#include "GPUlab.h"
#include "BoundaryConditions.h"


class Sim_SteadyStateMPI : public Simulation
{
    protected:

        const bool isroot;

        // simulation parameter
        double t, tend, tnextdump, dumpinterval, CFL;
        uint_t step, nsteps, nslices, saveinterval, fcount;
        int verbosity;
        bool restart, dryrun;
        char fname[256];

        // MPI cartesian grid extent
        uint_t npex, npey, npez;

        // main ingredients
        GridMPI *mygrid;
        GPUlab  *myGPU;

        // helper
        ArgumentParser parser;
        Profiler& profiler;

        virtual void _setup();
        virtual void _allocGPU();
        virtual void _ic();

        virtual void _dump(const std::string basename = "data");

        void _save();
        bool _restart();


    public:

        Sim_SteadyStateMPI(const int argc, const char ** argv, const int isroot);
        ~Sim_SteadyStateMPI()
        {
            delete mygrid;
            delete myGPU;
        }

        virtual void run();
};


class GPUlabSteadyState : public GPUlab
{
    protected:
        void _apply_bc(const double t = 0)
        {
            BoundaryConditions<GridMPI> bc(grid.pdata());
            if (myFeature[0] == SKIN) bc.template applyBC_reflecting<0,0,ghostmap::X>(halox.left);
            if (myFeature[1] == SKIN) bc.template applyBC_reflecting<0,1,ghostmap::X>(halox.right);
            if (myFeature[2] == SKIN) bc.template applyBC_reflecting<1,0,ghostmap::Y>(haloy.left);
            if (myFeature[3] == SKIN) bc.template applyBC_reflecting<1,1,ghostmap::Y>(haloy.right);
            if (myFeature[4] == SKIN) bc.template applyBC_reflecting<2,0,ghostmap::Z>(haloz.left);
            if (myFeature[5] == SKIN) bc.template applyBC_reflecting<2,1,ghostmap::Z>(haloz.right);
        }

    public:
        GPUlabSteadyState(GridMPI& grid, const uint_t nslices, const int verb) : GPUlab(grid, nslices, verb) { }
};

/* *
 * Sim_2DSBIMPI.h
 *
 * Created by Fabian Wermelinger on 7/19/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include "Sim_SteadyStateMPI.h"


class Sim_2DSBIMPI : public Sim_SteadyStateMPI
{
    uint_t dims[3];

    protected:

    virtual void _allocGPU();
    virtual void _ic();

    public:

    Sim_2DSBIMPI(const int argc, const char ** argv, const int isroot);
};


class GPUlab2DSBI_xreflect : public GPUlab
{
    protected:
        void _apply_bc(const double t = 0)
        {
            BoundaryConditions<GridMPI> bc(grid.pdata());
            if (myFeature[0] == SKIN) bc.template applyBC_reflecting<0,0,ghostmap::X>(halox.left);
            if (myFeature[1] == SKIN) bc.template applyBC_reflecting<0,1,ghostmap::X>(halox.right);
            if (myFeature[2] == SKIN) bc.template applyBC_absorbing <1,0,ghostmap::Y>(haloy.left);
            if (myFeature[3] == SKIN) bc.template applyBC_absorbing <1,1,ghostmap::Y>(haloy.right);
            if (myFeature[4] == SKIN) bc.template applyBC_absorbing <2,0,ghostmap::Z>(haloz.left);
            if (myFeature[5] == SKIN) bc.template applyBC_absorbing <2,1,ghostmap::Z>(haloz.right);
        }

    public:
        GPUlab2DSBI_xreflect(GridMPI& grid, const uint_t nslices, const int verb) : GPUlab(grid, nslices, verb) { }
};


class GPUlab2DSBI_yreflect : public GPUlab
{
    protected:
        void _apply_bc(const double t = 0)
        {
            BoundaryConditions<GridMPI> bc(grid.pdata());
            if (myFeature[0] == SKIN) bc.template applyBC_absorbing <0,0,ghostmap::X>(halox.left);
            if (myFeature[1] == SKIN) bc.template applyBC_absorbing <0,1,ghostmap::X>(halox.right);
            if (myFeature[2] == SKIN) bc.template applyBC_reflecting<1,0,ghostmap::Y>(haloy.left);
            if (myFeature[3] == SKIN) bc.template applyBC_reflecting<1,1,ghostmap::Y>(haloy.right);
            if (myFeature[4] == SKIN) bc.template applyBC_absorbing <2,0,ghostmap::Z>(haloz.left);
            if (myFeature[5] == SKIN) bc.template applyBC_absorbing <2,1,ghostmap::Z>(haloz.right);
        }

    public:
        GPUlab2DSBI_yreflect(GridMPI& grid, const uint_t nslices, const int verb) : GPUlab(grid, nslices, verb) { }
};


class GPUlab2DSBI_zreflect : public GPUlab
{
    protected:
        void _apply_bc(const double t = 0)
        {
            BoundaryConditions<GridMPI> bc(grid.pdata());
            if (myFeature[0] == SKIN) bc.template applyBC_absorbing <0,0,ghostmap::X>(halox.left);
            if (myFeature[1] == SKIN) bc.template applyBC_absorbing <0,1,ghostmap::X>(halox.right);
            if (myFeature[2] == SKIN) bc.template applyBC_absorbing <1,0,ghostmap::Y>(haloy.left);
            if (myFeature[3] == SKIN) bc.template applyBC_absorbing <1,1,ghostmap::Y>(haloy.right);
            if (myFeature[4] == SKIN) bc.template applyBC_reflecting<2,0,ghostmap::Z>(haloz.left);
            if (myFeature[5] == SKIN) bc.template applyBC_reflecting<2,1,ghostmap::Z>(haloz.right);
        }

    public:
        GPUlab2DSBI_zreflect(GridMPI& grid, const uint_t nslices, const int verb) : GPUlab(grid, nslices, verb) { }
};

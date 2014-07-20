/* *
 * LSRK3_IntegratorMPI.h
 *
 * Created by Fabian Wermelinger on 7/20/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include "GridMPI.h"
#include "GPUlab.h"


class LSRK3_IntegratorMPI
{
    const double h, CFL;
    double dt;
    float sos;
    const int verbosity;

    const GridMPI *grid;
    GPUlab *GPU;

    public:

        LSRK3_IntegratorMPI(const GridMPI *grid_, GPUlab *GPU_, const double CFL_, const int verb) :
            grid(grid_), GPU(GPU_), h(grid_->getH()), CFL(CFL_), verbosity(verb) { }

        double operator()(const double dt_max);
};

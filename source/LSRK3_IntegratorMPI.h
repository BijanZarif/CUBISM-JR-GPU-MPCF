/* *
 * LSRK3_IntegratorMPI.h
 *
 * Created by Fabian Wermelinger on 7/20/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include "Types.h"
#include "GridMPI.h"
#include "GPUlab.h"
#include "ArgumentParser.h"
#include <string>

class LSRK3_IntegratorMPI
{
    const double h, CFL;
    double dt;
    float sos;
    int verbosity;
    std::string SOSkernel;

    const GridMPI *grid;
    GPUlab *GPU;


    public:

    LSRK3_IntegratorMPI(const GridMPI *grid_, GPUlab *GPU_, const double CFL_, ArgumentParser& parser) :
        grid(grid_), GPU(GPU_), h(grid_->getH()), CFL(CFL_)
    {
        verbosity = parser("-verb").asInt(0);
        SOSkernel = parser("-SOSkernel").asString("cuda");
    }

    double operator()(const double dt_max);
};

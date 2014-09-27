/* *
 * LSRK3_IntegratorMPI.h
 *
 * Created by Fabian Wermelinger on 7/20/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include "Types.h"
#include "GridMPI.h"
#include "GPUlabMPI.h"
#include "ArgumentParser.h"
#include <string>

class LSRK3_IntegratorMPI
{
    const double h, CFL;
    double dt;
    float sos;
    int verbosity;
    std::string SOSkernel, UPkernel;
    const bool isroot;

    const GridMPI* grid;
    GPUlabMPI* GPU;

    template <typename Kupdate>
    double _RKstepGPU(const Real dtinvh)
    {
        double trk1, trk2, trk3;
        {// stage 1
            GPU->load_ghosts();
            trk1 = GPU->template process_all<Kupdate>(0, 1./4, dtinvh);
            if (isroot && verbosity) printf("RK stage 1 takes %f sec\n", trk1);
        }
        {// stage 2
            GPU->load_ghosts();
            trk2 = GPU->template process_all<Kupdate>(-17./32, 8./9, dtinvh);
            if (isroot && verbosity) printf("RK stage 2 takes %f sec\n", trk2);
        }
        {// stage 3
            GPU->load_ghosts();
            trk3 = GPU->template process_all<Kupdate>(-32./27, 3./4, dtinvh);
            if (isroot && verbosity) printf("RK stage 3 takes %f sec\n", trk3);
        }
        return trk1 + trk2 + trk3;
    }

public:

    LSRK3_IntegratorMPI(const GridMPI *grid_, GPUlabMPI *GPU_, const double CFL_, ArgumentParser& parser, const bool isroot_=true) :
        grid(grid_), GPU(GPU_), h(grid_->getH()), CFL(CFL_), isroot(isroot_)
    {
        verbosity = parser("-verb").asInt(0);
        SOSkernel = parser("-SOSkernel").asString("cuda");
        UPkernel  = parser("-UPkernel").asString("cpp");
    }

    double operator()(const double dt_max);
};

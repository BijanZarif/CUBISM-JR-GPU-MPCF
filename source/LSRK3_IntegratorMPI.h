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
        // Williamson
        /* const Real A1 = 0.0; */
        /* const Real A2 = -17./32; */
        /* const Real A3 = -32./27; */
        /* const Real B1 = 1./4; */
        /* const Real B2 = 8./9; */
        /* const Real B3 = 3./4; */

        // Gottlieb & Shu
        const Real A1 = 0.0;
        const Real A2 = -2.915492524638791;
        const Real A3 = -0.000000093517376;
        const Real B1 = 0.924574000000000;
        const Real B2 = 0.287713063186749;
        const Real B3 = 0.626538109512740;

        double trk1, trk2, trk3;
        {// stage 1
            GPU->load_ghosts();
            trk1 = GPU->template process_all<Kupdate>(A1, B1, dtinvh);
            if (isroot && verbosity) printf("RK stage 1 takes %f sec\n", trk1);
        }
        {// stage 2
            GPU->load_ghosts();
            trk2 = GPU->template process_all<Kupdate>(A2, B2, dtinvh);
            if (isroot && verbosity) printf("RK stage 2 takes %f sec\n", trk2);
        }
        {// stage 3
            GPU->load_ghosts();
            trk3 = GPU->template process_all<Kupdate>(A3, B3, dtinvh);
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

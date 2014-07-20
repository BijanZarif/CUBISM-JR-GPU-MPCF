/* *
 * LSRK3_IntegratorMPI.cpp
 *
 * Created by Fabian Wermelinger on 7/20/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <cassert>
#include <stdio.h>
#include <mpi.h>
#include "LSRK3_Integrator.h"

#include "MaxSpeedOfSound.h"

double LSRK3_IntegratorMPI::operator()(const double dt_max)
{
    const double tsos = GPU->max_sos(sos);
    /* const double tsos = _maxSOS<MaxSpeedOfSound_CPP>(sos); */
    assert(sos > 0);
    if (verbosity) printf("sos = %f (took %f sec)\n", sos, tsos);

    dt = CFL*h/sos;
    dt = dt_max < dt ? dt_max : dt;

    MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, grid->getCartComm());

    // 2.) Compute RHS and update using LSRK3
    double trk1, trk2, trk3;
    {// stage 1
        GPU->load_ghosts();
        trk1 = GPU->process_all(0, 1./4, dt/h);
        if (verbosity) printf("RK stage 1 takes %f sec\n", trk1);
    }
    {// stage 2
        GPU->load_ghosts();
        trk2 = GPU->process_all(-17./32, 8./9, dt/h);
        if (verbosity) printf("RK stage 2 takes %f sec\n", trk2);
    }
    {// stage 3
        GPU->load_ghosts();
        trk3 = GPU->process_all(-32./27, 3./4, dt/h);
        if (verbosity) printf("RK stage 3 takes %f sec\n", trk3);
    }
    if (verbosity) printf("netto step takes %f sec\n", tsos + trk1 + trk2 + trk3);

    return dt;
}

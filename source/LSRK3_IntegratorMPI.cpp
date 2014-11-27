/* *
 * LSRK3_IntegratorMPI.cpp
 *
 * Created by Fabian Wermelinger on 7/20/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "LSRK3_IntegratorMPI.h"
#include "Timer.h"
#include "MaxSpeedOfSound.h"
#include "Update_CPP.h"

#ifdef _QPXEMU_
#include "MaxSpeedOfSound_QPX.h"
#include "Update_QPX.h"
#endif

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


// main state variables
double LSRK3_DataMPI::time = 0.0;
size_t LSRK3_DataMPI::step = 0;

// LSRK3 coeffs, default to Gottlieb & Shu -> see constructor
// LSRK3_IntegratorMPI
Real LSRK3_DataMPI::A1 = 0.0;
Real LSRK3_DataMPI::A2 = 0.0;
Real LSRK3_DataMPI::A3 = 0.0;
Real LSRK3_DataMPI::B1 = 0.0;
Real LSRK3_DataMPI::B2 = 0.0;
Real LSRK3_DataMPI::B3 = 0.0;

Histogram LSRK3_DataMPI::histogram;
size_t LSRK3_DataMPI::ReportFreq = 1;
double LSRK3_DataMPI::t_RHS = 0.0;
double LSRK3_DataMPI::t_UPDATE = 0.0;
double LSRK3_DataMPI::t_COMM = 0.0;


// tiny helper
template <typename KSOS>
static double _maxSOS(const GridMPI * const grid, float& sos)
{
    KSOS kernel;
    Timer tsos;
    tsos.start();
    sos = kernel.compute(grid->pdata());
    return tsos.stop();
}


double LSRK3_IntegratorMPI::operator()(const double dt_max)
{
    double tsos;
    if (SOSkernel == "cuda") tsos = GPU->max_sos(sos);
    else if (SOSkernel == "cpp") tsos = _maxSOS<MaxSpeedOfSound_CPP>(grid, sos);
#ifdef _QPXEMU_
    else if (SOSkernel == "qpx") tsos = _maxSOS<MaxSpeedOfSound_QPX>(grid, sos);
#endif
    else
    {
        fprintf(stderr, "SOS kernel = %s... Not supported!\n", SOSkernel.c_str());
        abort();
    }
    assert(sos > 0);

    MPI_Allreduce(MPI_IN_PLACE, &sos, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    if (isroot) printf("Max SOS = %f (took %f sec)\n", sos, tsos);

    dt = CFL * h / sos;
    dt = dt_max < dt ? dt_max : dt;

    /* MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); */

    // update state
    LSRK3_DataMPI::time += dt;
    LSRK3_DataMPI::step++;

    // 2.) Compute RHS and update using LSRK3
    double trk3;
    if (UPkernel == "cpp") trk3 = _RKstepGPU<Update_CPP>(dt/h);
#ifdef _QPXEMU_
    else if (UPkernel == "qpx") trk3 = _RKstepGPU<Update_QPX>(dt/h);
#endif
    else
    {
        fprintf(stderr, "UP kernel = %s... Not supported!\n", UPkernel.c_str());
        abort();
    }

    if (isroot) printf("netto step takes %f sec\n", tsos + trk3);

    return dt;
}

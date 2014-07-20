/* *
 * Sim_SteadyStateMPI.cpp
 *
 * Created by Fabian Wermelinger on 7/18/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <cassert>
#include <omp.h>
#include <string>
#include <fstream>
#include <iomanip>

#include "Sim_SteadyStateMPI.h"
#include "LSRK3_IntegratorMPI.h"
#include "HDF5Dumper_MPI.h"
/* #include "SerializerIO_WaveletCompression_MPI_Simple.h" */

using namespace std;


Sim_SteadyStateMPI::Sim_SteadyStateMPI(const int argc, const char ** argv, const int isroot_)
    : isroot(isroot_), t(0.0), step(0), fcount(0), mygrid(NULL), myGPU(NULL), parser(argc, argv)
{ }


void Sim_SteadyStateMPI::_setup()
{
    // parse mandatory arguments
    parser.set_strict_mode();
    tend         = parser("-tend").asDouble();
    CFL          = parser("-cfl").asDouble();
    nslices      = parser("-nslices").asInt();
    dumpinterval = parser("-dumpinterval").asDouble();
    saveinterval = parser("-saveinterval").asInt();
    parser.unset_strict_mode();

    // parse optional aruments
    verbosity = parser("-verb").asInt(0);
    restart   = parser("-restart").asBool(false);
    nsteps    = parser("-nsteps").asInt(0);

    // MPI
    npex = parser("-npex").asInt(1);
    npey = parser("-npey").asInt(1);
    npez = parser("-npez").asInt(1);

    // assign dependent stuff
    tnextdump = dumpinterval;
    mygrid    = new GridMPI(npex, npey, npez);
    assert(mygrid != NULL);

    // setup initial condition
    if (restart)
    {
        if(_restart())
        {
            printf("Restarting at step %d, physical time %f\n", step, t);
            _dump("restart_ic");
            tnextdump = (fcount+1)*dumpinterval;
        }
        else
        {
            printf("Loading restart file was not successful... Abort\n");
            exit(1);
        }
    }
    else
    {
        _ic();
        _dump();
    }

    _allocGPU();
    assert(myGPU != NULL);
}


void Sim_SteadyStateMPI::_allocGPU()
{
    printf("Allocating GPUlabSteadyState...\n");
    myGPU = new GPUlabSteadyState(*mygrid, nslices, verbosity);
}


void Sim_SteadyStateMPI::_ic()
{
    printf("=====================================================================\n");
    printf("                            Steady State                             \n");
    printf("=====================================================================\n");
    // default initial condition
    const double r  = parser("-rho").asDouble(1.0);
    const double u  = parser("-u").asDouble(0.0);
    const double v  = parser("-v").asDouble(0.0);
    const double w  = parser("-w").asDouble(0.0);
    const double p  = parser("-p").asDouble(1.0);
    const double g  = parser("-g").asDouble(1.4);
    const double pc = parser("-pc").asDouble(0.0);

    const double ru = r*u;
    const double rv = r*v;
    const double rw = r*w;
    const double G  = 1.0/(g - 1.0);
    const double P  = g*pc*G;
    const double e  = G*p + P + 0.5*r*(u*u + v*v + w*w);

    typedef GridMPI::PRIM var;
    GridMPI& icgrid = *mygrid;

#pragma omp paralell for
    for (int iz = 0; iz < GridMPI::sizeZ; ++iz)
        for (int iy = 0; iy < GridMPI::sizeY; ++iy)
            for (int ix = 0; ix < GridMPI::sizeX; ++ix)
            {
                icgrid(ix, iy, iz, var::R) = r;
                icgrid(ix, iy, iz, var::U) = ru;
                icgrid(ix, iy, iz, var::V) = rv;
                icgrid(ix, iy, iz, var::W) = rw;
                icgrid(ix, iy, iz, var::E) = e;
                icgrid(ix, iy, iz, var::G) = G;
                icgrid(ix, iy, iz, var::P) = P;
            }
}


void Sim_SteadyStateMPI::_dump(const string basename)
{
    const string dump_path = parser("-fpath").asString(".");

    sprintf(fname, "%s_%04d", basename.c_str(), fcount++);
    printf("Dumping file %s at step %d, time %f\n", fname, step, t);
    DumpHDF5_MPI<GridMPI, myTensorialStreamer>(*mygrid, step, fname, dump_path);
}


void Sim_SteadyStateMPI::_save()
{
    const string dump_path = parser("-fpath").asString(".");

    if (isroot)
    {
        ofstream saveinfo("save.info");
        saveinfo << setprecision(15) << t << endl;
        saveinfo << step << endl;
        saveinfo << fcount << endl;
        saveinfo.close();
    }
    DumpHDF5_MPI<GridMPI, mySaveStreamer>(*mygrid, step, "save.data", dump_path);
}


bool Sim_SteadyStateMPI::_restart()
{
    const string dump_path = parser("-fpath").asString(".");

    ifstream saveinfo("save.info");
    if (saveinfo.good())
    {
        saveinfo >> t;
        saveinfo >> step;
        saveinfo >> fcount;
        ReadHDF5_MPI<GridMPI, mySaveStreamer>(*mygrid, "save.data", dump_path);
        return true;
    }
    else
        return false;
}


void Sim_SteadyStateMPI::run()
{
    _setup();

    double dt, dt_max;
    LSRK3_IntegratorMPI * const stepper = new LSRK3_IntegratorMPI(mygrid, myGPU, CFL, verbosity);

    while (t < tend)
    {
        dt_max = (tend-t) < (tnextdump-t) ? (tend-t) : (tnextdump-t);
        dt = (*stepper)(dt_max);

        t += dt;
        ++step;

        printf("step id is %d, physical time %f (dt = %f)\n", step, t, dt);

        if ((float)t == (float)tnextdump)
        {
            tnextdump += dumpinterval;
            _dump();
        }
        /* if (step % 10 == 0) _dump(); */

        if (step % saveinterval == 0)
        {
            printf("Saving time step...\n");
            _save();
        }

        if (step == nsteps) break;
    }
    _dump();
    delete stepper;
}

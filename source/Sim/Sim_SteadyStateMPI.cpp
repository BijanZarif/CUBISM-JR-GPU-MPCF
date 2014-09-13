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
#include "HDF5Dumper_MPI.h"
/* #include "SerializerIO_WaveletCompression_MPI_Simple.h" */

using namespace std;


Sim_SteadyStateMPI::Sim_SteadyStateMPI(const int argc, const char ** argv, const int isroot_)
    : isroot(isroot_), t(0.0), step(0), fcount(0), mygrid(NULL), myGPU(NULL), parser(argc, argv), profiler(GPUlab::get_profiler())
{ }


void Sim_SteadyStateMPI::_setup()
{
    dryrun = parser("-dryrun").asBool(false);

    dumpinterval = HUGE_VAL;
    if (!dryrun)
    {
        // parse mandatory arguments
        parser.set_strict_mode();
        tend         = parser("-tend").asDouble();
        CFL          = parser("-cfl").asDouble();
        nslices      = parser("-nslices").asInt();
        dumpinterval = parser("-dumpinterval").asDouble();
        saveperiod   = parser("-saveperiod").asInt();
        parser.unset_strict_mode();
    }

    // with IO
    bIO = parser("-IO").asBool(true);

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

    // allocate GPU
    if (!dryrun)
    {
        _allocGPU();
        assert(myGPU != NULL);
    }
    else
        if (isroot) printf("No GPU allocated...\n");

    // allocate integrator
    stepper = new LSRK3_IntegratorMPI(mygrid, myGPU, CFL, parser);

    // setup initial condition
    if (restart)
    {
        if(_restart())
        {
            if (isroot) printf("Restarting at step %d, physical time %f\n", step, t);
            --fcount; // last dump before restart condition incremented fcount.
            // Decrement by one and dump restart IC, which increments fcount
            // again to start with the correct count.
            if (bIO) _dump("restart_ic");
        }
        else
        {
            printf("Loading restart file was not successful... Abort\n");
            abort();
        }
    }
    else
    {
        _ic();
        if (bIO) _dump();
    }
}


void Sim_SteadyStateMPI::_allocGPU()
{
    if (isroot) printf("Allocating GPUlabSteadyState...\n");
    myGPU = new GPUlabSteadyState(*mygrid, nslices, verbosity);
}


void Sim_SteadyStateMPI::_ic()
{
    if (isroot)
    {
        printf("=====================================================================\n");
        printf("                            Steady State                             \n");
        printf("=====================================================================\n");
    }
    // default initial condition
    const double r  = parser("-rho").asDouble(1.0);
    const double u  = parser("-u").asDouble(0.0);
    const double v  = parser("-v").asDouble(0.0);
    const double w  = parser("-w").asDouble(0.0);
    const double p  = parser("-p").asDouble(1.0);
    const double g  = parser("-g").asDouble(1.4);
    const double pc = parser("-pc").asDouble(0.0);
    /* const double r  = parser("-rho").asDouble(1.5); */
    /* const double u  = parser("-u").asDouble(1.0); */
    /* const double v  = parser("-v").asDouble(1.0); */
    /* const double w  = parser("-w").asDouble(1.0); */
    /* const double p  = parser("-p").asDouble(1.0); */
    /* const double g  = parser("-g").asDouble(1.5); */
    /* const double pc = parser("-pc").asDouble(1.0); */
    MaterialDictionary::gamma1 = g;
    MaterialDictionary::pc1 = pc;

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

    sprintf(fname, "%s_%04d", basename.c_str(), fcount);
    if (isroot) printf("Dumping file %s at step %d, time %f\n", fname, step, t);
    DumpHDF5_MPI<GridMPI, myTensorialStreamer>(*mygrid, step, fname, dump_path);
    ++fcount;
}


void Sim_SteadyStateMPI::_save()
{
    const string dump_path = parser("-fpath").asString(".");

    if (isroot)
    {
        ofstream saveinfo("save.info");
        saveinfo << setprecision(16) << scientific << t << endl;
        saveinfo << step << endl;
        saveinfo << fcount << endl;
        saveinfo << setprecision(16) << scientific << (tnextdump - dumpinterval) << endl; // last dump time
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
        double last_dumptime;
        saveinfo >> last_dumptime;
        tnextdump = last_dumptime + dumpinterval;
        ReadHDF5_MPI<GridMPI, mySaveStreamer>(*mygrid, "save.data", dump_path);
        // since t >= last_dumptime and dumpinterval might be anything new:
        while (t > tnextdump)
            tnextdump += dumpinterval;
        return true;
    }
    else
        return false;
}


void Sim_SteadyStateMPI::run()
{
    _setup();

    if (dryrun)
    {
        if (isroot) printf("Dry Run...\n");
        return;
    }
    else
    {
        double dt, dt_max;

        const uint_t step_start = step; // such that -nsteps is a relative measure
        while (t < tend)
        {
            profiler.push_start("EVOLVE");
            dt_max = (tend-t) < (tnextdump-t) ? (tend-t) : (tnextdump-t);
            dt = (*stepper)(dt_max);
            profiler.pop_stop();

            t += dt;
            ++step;

            if (isroot) printf("step id is %d, physical time %e (dt = %e)\n", step, t, dt);

            if (bIO && (float)t == (float)tnextdump)
            {
                profiler.push_start("DUMP");
                tnextdump += dumpinterval;
                _dump();
                profiler.pop_stop();
            }
            /* if (bIO && step % 10 == 0) _dump(); */

            if (step % saveperiod == 0)
            {
                if (isroot) printf("Saving time step...\n");
                _save();
            }

            if (step % 10 == 0)
                profiler.printSummary();

            if ((step-step_start) == nsteps) break;
        }

        profiler.printSummary();

        if (bIO) _dump();

        return;
    }
}

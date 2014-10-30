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
#include <cmath>

#include "Sim_SteadyStateMPI.h"
#include "HDF5Dumper_MPI.h"
/* #include "SerializerIO_WaveletCompression_MPI_Simple.h" */

using namespace std;


Sim_SteadyStateMPI::Sim_SteadyStateMPI(const int argc, const char ** argv, const int isroot_)
    : isroot(isroot_), fcount(0), mygrid(NULL), myGPU(NULL), parser(argc, argv), profiler(GPUlabMPI::get_profiler())
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
    bHDF= parser("-HDF").asBool(true);

    // parse optional aruments
    verbosity = parser("-verb").asInt(0);
    restart   = parser("-restart").asBool(false);
    nsteps    = parser("-nsteps").asInt(0);

    // MPI processes arranged on cartesian grid
    npex = parser("-npex").asInt(1);
    npey = parser("-npey").asInt(1);
    npez = parser("-npez").asInt(1);

    // maximum domain extent
    maxextent = parser("-maxextent").asDouble(1.0);

    // assign dependent stuff
    tnextdump = dumpinterval;
    mygrid    = new GridMPI(npex, npey, npez, maxextent);
    assert(mygrid != NULL);

    // smooth length
    SimTools::EPSILON = static_cast<Real>(parser("-mollfactor").asDouble(1)) * std::sqrt(3.) * mygrid->getH();

    // allocate GPU
    if (!dryrun)
    {
        _allocGPU();
        assert(myGPU != NULL);
    }
    else
        if (isroot) printf("No GPU allocated...\n");

    // allocate integrator
    stepper = new LSRK3_IntegratorMPI(mygrid, myGPU, CFL, parser, isroot);

    // setup initial condition
    if (restart)
    {
        if(_restart())
        {
            if (isroot) printf("Restarting at step %d, physical time %f\n", LSRK3_DataMPI::step, LSRK3_DataMPI::time);
            --fcount; // last dump before restart condition incremented fcount.
            // Decrement by one and dump restart IC, which increments fcount
            // again to start with the correct count.
            if (bIO) _dump("restart_ic");
        }
        else
        {
            if (isroot) printf("Loading restart file was not successful... Abort\n");
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
    if (isroot) printf("Allocating GPUlabMPISteadyState...\n");
    myGPU = new GPUlabMPISteadyState(*mygrid, nslices, verbosity);
}


void Sim_SteadyStateMPI::_set_constants()
{
    if (isroot)
    {
        printf("=====================================================================\n");
        printf("                            Steady State                             \n");
        printf("=====================================================================\n");
    }
}

void Sim_SteadyStateMPI::_ic()
{
    _set_constants();

    // default initial condition
    const double r  = parser("-rho").asDouble(1.0);
    const double u  = parser("-u").asDouble(0.0);
    const double v  = parser("-v").asDouble(0.0);
    const double w  = parser("-w").asDouble(0.0);
    const double p  = parser("-p").asDouble(1.0);
    const double g  = parser("-g").asDouble(1.4);
    const double pc = parser("-pc").asDouble(0.0);
    MaterialDictionary::rho1 = r;
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
    if (isroot) printf("Dumping file %s at step %d, time %f\n", fname, LSRK3_DataMPI::step, LSRK3_DataMPI::time);
    DumpHDF5_MPI<GridMPI, myTensorialStreamer>(*mygrid, LSRK3_DataMPI::step, fname, dump_path);
    ++fcount;
}


void Sim_SteadyStateMPI::_save()
{
    const string dump_path = parser("-fpath").asString(".");

    if (isroot)
    {
        ofstream saveinfo("save.info");
        saveinfo << setprecision(16) << scientific << LSRK3_DataMPI::time << endl;
        saveinfo << LSRK3_DataMPI::step << endl;
        saveinfo << fcount << endl;
        saveinfo << setprecision(16) << scientific << (tnextdump - dumpinterval) << endl; // last dump time
        saveinfo.close();
    }
    /* DumpHDF5_MPI<GridMPI, mySaveStreamer>(*mygrid, LSRK3_DataMPI::step, "save.data", dump_path); */

    // save in 4 parts to allow data sets exceeding 2GB for a single process
    // http://www.hdfgroup.org/hdf5-quest.html#p2gb
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part1>(*mygrid, LSRK3_DataMPI::step, "save.data.part1", dump_path);
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part2>(*mygrid, LSRK3_DataMPI::step, "save.data.part2", dump_path);
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part3>(*mygrid, LSRK3_DataMPI::step, "save.data.part3", dump_path);
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part4>(*mygrid, LSRK3_DataMPI::step, "save.data.part4", dump_path);
}


bool Sim_SteadyStateMPI::_restart()
{
    const string dump_path = parser("-fpath").asString(".");

    ifstream saveinfo("save.info");
    if (saveinfo.good())
    {
        saveinfo >> LSRK3_DataMPI::time;
        saveinfo >> LSRK3_DataMPI::step;
        saveinfo >> fcount;
        double last_dumptime;
        saveinfo >> last_dumptime;
        tnextdump = last_dumptime + dumpinterval;
        /* ReadHDF5_MPI<GridMPI, mySaveStreamer>(*mygrid, "save.data", dump_path); */

        // 4 parts
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part1>(*mygrid, "save.data.part1", dump_path);
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part2>(*mygrid, "save.data.part2", dump_path);
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part3>(*mygrid, "save.data.part3", dump_path);
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part4>(*mygrid, "save.data.part4", dump_path);

        // make sure tnextdump > time
        while (LSRK3_DataMPI::time > tnextdump)
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

        const uint_t step_start = LSRK3_DataMPI::step; // such that -nsteps is a relative measure
        while (LSRK3_DataMPI::time < tend)
        {
            const double dt_final = tend - LSRK3_DataMPI::time;
            const double dt_dump  = tnextdump - LSRK3_DataMPI::time;

            // here is where the stuff happens
            profiler.push_start("EVOLVE");
            dt_max = dt_final < dt_dump ? dt_final : dt_dump;
            dt = (*stepper)(dt_max); // step ahead
            profiler.pop_stop();

            // post processings
            if (isroot) printf("step id is %d, physical time %e (dt = %e)\n", LSRK3_DataMPI::step, LSRK3_DataMPI::time, dt);

            if (bIO && (float)LSRK3_DataMPI::time == (float)tnextdump)
            {
                profiler.push_start("DUMP");
                tnextdump += dumpinterval;
                _dump();
                profiler.pop_stop();
            }
            /* if (bIO && LSRK3_DataMPI::step % 10 == 0) _dump(); */

            if (LSRK3_DataMPI::step % saveperiod == 0)
            {
                if (isroot) printf("Saving time step...\n");
                _save();
            }

            if (isroot && LSRK3_DataMPI::step % 10 == 0)
                profiler.printSummary();

            if ((LSRK3_DataMPI::step - step_start) == nsteps) break;
        }

        if (isroot) profiler.printSummary();

        if (bIO) _dump();

        return;
    }
}

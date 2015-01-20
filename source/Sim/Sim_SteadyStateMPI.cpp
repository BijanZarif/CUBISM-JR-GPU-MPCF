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
#include <cstdio>
#include <iomanip>
#include <cmath>

#include "Sim_SteadyStateMPI.h"
#include "HDF5Dumper_MPI.h"

using namespace std;

Real Sim_SteadyStateData::r = 0.0;
Real Sim_SteadyStateData::u = 0.0;
Real Sim_SteadyStateData::v = 0.0;
Real Sim_SteadyStateData::w = 0.0;
Real Sim_SteadyStateData::e = 0.0;
Real Sim_SteadyStateData::G = 0.0;
Real Sim_SteadyStateData::P = 0.0;


Sim_SteadyStateMPI::Sim_SteadyStateMPI(const int argc, const char ** argv, const int isroot_)
    : isroot(isroot_), fcount(0), save_flipflop(0), mygrid(NULL), myGPU(NULL), parser(argc, argv), profiler(GPUlabMPI::get_profiler())
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
        dumpinterval = tnextdump = parser("-dumpinterval").asDouble();
        saveperiod   = parser("-saveperiod").asInt();
        parser.unset_strict_mode();
    }

    // with IO
    bIO = parser("-IO").asBool(true);
    bHDF= parser("-HDF").asBool(true);
    bVP = parser("-VP").asBool(false);

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

    // init data
    mygrid = new GridMPI(npex, npey, npez, maxextent);
    assert(mygrid != NULL);

    // we use artifical subblocks that are smaller than the main block
    parser.set_strict_mode();
    const int subcells_x = parser("-subcellsX").asInt();
    parser.unset_strict_mode();
    const int subcells_y = parser("-subcellsY").asInt(subcells_x);
    const int subcells_z = parser("-subcellsZ").asInt(subcells_x);

    // wavelet dumps only work if subcells_x = subcells_y = subcells_z =
    // _SUBBLOCKSIZE_
    if (subcells_x != _SUBBLOCKSIZE_ && subcells_y != _SUBBLOCKSIZE_ && subcells_z !=_SUBBLOCKSIZE_)
    {
        bVP = false;
        if (isroot) printf("WARNING: VP dumps disabled due to dimension mismatch!\n");
    }
    // set up artificial mesh
    subblocks.make_submesh(mygrid, subcells_x, subcells_y, subcells_z);

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
            if (isroot) printf("Restarting at step %d, time %f\n", LSRK3_DataMPI::step, LSRK3_DataMPI::time);
            if (fcount > 0) --fcount;
            if (bIO) _take_a_dump("restart_ic");
        }
        else
        {
            if (isroot) printf("Restart is not ok! Can't find save.info file... Abort\n");
            abort();
        }
    }
    else
    {
        _ic();
        if (bIO) _take_a_dump();
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
    Sim_SteadyStateData::r = r;
    Sim_SteadyStateData::u = ru;
    Sim_SteadyStateData::v = rv;
    Sim_SteadyStateData::w = rw;
    Sim_SteadyStateData::e = e;
    Sim_SteadyStateData::G = G;
    Sim_SteadyStateData::P = P;

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

    sprintf(fname, "%s_%05d", basename.c_str(), fcount);
    if (isroot) printf("Dumping HDF file %s at step %d, time %f\n", fname, LSRK3_DataMPI::step, LSRK3_DataMPI::time);
    DumpHDF5_MPI<GridMPI, myTensorialStreamer>(*mygrid, LSRK3_DataMPI::step, fname, dump_path);
    ++fcount;
}


void Sim_SteadyStateMPI::_vp(const std::string basename)
{
    if (isroot) cout << "Dumping MPI VP...\n" ;

    const string path = parser("-fpath").asString(".");

    stringstream streamer;
    streamer << path;
    streamer << "/";
    streamer << basename;
    streamer.setf(ios::dec | ios::right);
    streamer.width(5);
    streamer.fill('0');
    streamer << LSRK3_DataMPI::step;

    mywaveletdumper.verbose();
    mywaveletdumper.set_threshold(5e-3);
    mywaveletdumper.Write<4>(subblocks, streamer.str());
    mywaveletdumper.set_threshold(1e-3);
    mywaveletdumper.Write<5>(subblocks, streamer.str());
    //mywaveletdumper.Write<6>(subblocks, streamer.str());

    //used for debug
#if 0
    {
        //mywaveletdumper.force_close();
        if (isroot)
        {
            printf("\n\nREADING BEGINS===================\n");
            //just checking
            mywaveletdumper.Read(streamer.str());
            printf("\n\nREADING ENDS===================\n");
        }
    }
#endif
    if (isroot) cout << "done" << endl;
}


void Sim_SteadyStateMPI::_save()
{
    const string dump_path = parser("-fpath").asString(".");
    char fname[256];

    if (isroot)
    {
        sprintf(fname, "save.info%d", save_flipflop);
        ofstream saveinfo(fname);
        saveinfo << setprecision(16) << scientific << LSRK3_DataMPI::time << endl;
        saveinfo << LSRK3_DataMPI::step << endl;
        saveinfo << fcount << endl;
        saveinfo << setprecision(16) << scientific << (tnextdump - dumpinterval) << endl; // last dump time
        saveinfo << save_flipflop << endl;
        saveinfo.close();
    }
    /* DumpHDF5_MPI<GridMPI, mySaveStreamer>(*mygrid, LSRK3_DataMPI::step, "save.data", dump_path); */

    // save in 7 parts to allow data sets exceeding 2GB for a single process
    // http://www.hdfgroup.org/hdf5-quest.html#p2gb
    sprintf(fname, "save.data%d.part%d", save_flipflop, 1);
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part1>(*mygrid, LSRK3_DataMPI::step, fname, dump_path);
    sprintf(fname, "save.data%d.part%d", save_flipflop, 2);
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part2>(*mygrid, LSRK3_DataMPI::step, fname, dump_path);
    sprintf(fname, "save.data%d.part%d", save_flipflop, 3);
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part3>(*mygrid, LSRK3_DataMPI::step, fname, dump_path);
    sprintf(fname, "save.data%d.part%d", save_flipflop, 4);
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part4>(*mygrid, LSRK3_DataMPI::step, fname, dump_path);
    sprintf(fname, "save.data%d.part%d", save_flipflop, 5);
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part5>(*mygrid, LSRK3_DataMPI::step, fname, dump_path);
    sprintf(fname, "save.data%d.part%d", save_flipflop, 6);
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part6>(*mygrid, LSRK3_DataMPI::step, fname, dump_path);
    sprintf(fname, "save.data%d.part%d", save_flipflop, 7);
    DumpHDF5_MPI<GridMPI, mySaveStreamer_part7>(*mygrid, LSRK3_DataMPI::step, fname, dump_path);

    // next
    save_flipflop = 1 - save_flipflop;
}


bool Sim_SteadyStateMPI::_restart()
{
    const string dump_path = parser("-fpath").asString(".");
    char fname[256];

    // the file that is read must be named like this.  Note that there is no
    // flipflop information here, so use e.g. a symlink to specify the restart
    // data
    ifstream saveinfo("save.info");
    if (saveinfo.good())
    {
        saveinfo >> LSRK3_DataMPI::time;
        saveinfo >> LSRK3_DataMPI::step;
        saveinfo >> fcount;
        double last_dumptime;
        saveinfo >> last_dumptime;
        tnextdump = last_dumptime + dumpinterval;
        saveinfo >> save_flipflop;
        /* ReadHDF5_MPI<GridMPI, mySaveStreamer>(*mygrid, "save.data", dump_path); */

        // 7 parts
        sprintf(fname, "save.data%d.part%d", save_flipflop, 1);
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part1>(*mygrid, fname, dump_path);
        sprintf(fname, "save.data%d.part%d", save_flipflop, 2);
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part2>(*mygrid, fname, dump_path);
        sprintf(fname, "save.data%d.part%d", save_flipflop, 3);
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part3>(*mygrid, fname, dump_path);
        sprintf(fname, "save.data%d.part%d", save_flipflop, 4);
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part4>(*mygrid, fname, dump_path);
        sprintf(fname, "save.data%d.part%d", save_flipflop, 5);
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part5>(*mygrid, fname, dump_path);
        sprintf(fname, "save.data%d.part%d", save_flipflop, 6);
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part6>(*mygrid, fname, dump_path);
        sprintf(fname, "save.data%d.part%d", save_flipflop, 7);
        ReadHDF5_MPI<GridMPI, mySaveStreamer_part7>(*mygrid, fname, dump_path);

        // dont overwrite this save data for next save
        save_flipflop = 1 - save_flipflop;

        // make sure tnextdump > time
        while (LSRK3_DataMPI::time > tnextdump)
            tnextdump += dumpinterval;
        return true;
    }
    else
        return false;
}


void Sim_SteadyStateMPI::_take_a_dump(const string basename)
{
    if (bHDF)
    {
        profiler.push_start("HDF DUMP");
        _dump(basename);
        profiler.pop_stop();
    }

    if (bVP)
    {
        profiler.push_start("VP DUMP");
        _vp( basename + "wavelet" );
        profiler.pop_stop();
    }
}


void Sim_SteadyStateMPI::run()
{
    _setup();

    if (parser("-printargs").asBool(true) && isroot) parser.print_options();

#ifndef _FLOAT_PRECISION_
    bVP = false;
    if (isroot) printf("WARNING: VP dumps disabled due to double precision run!\n");
#endif

    // log dumps
    FILE* fp;
    if (bIO) fp = fopen("dump.log", "a");

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
            dt_max = dt_final;
            if (bIO) dt_max = dt_max < dt_dump ? dt_max : dt_dump;
            dt = (*stepper)(dt_max); // step ahead
            profiler.pop_stop();

            // post processings
            if (isroot) printf("step id is %d, time %e (dt = %e)\n", LSRK3_DataMPI::step, LSRK3_DataMPI::time, dt);

            if (bIO && (float)LSRK3_DataMPI::time == (float)tnextdump)
            {
                fprintf(fp, "step=%d\ttime=%e\n", LSRK3_DataMPI::step, LSRK3_DataMPI::time);
                tnextdump += dumpinterval;
                _take_a_dump();
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

        if (bIO)
        {
            // dump final data
            fprintf(fp, "step=%d\ttime=%e\n", LSRK3_DataMPI::step, LSRK3_DataMPI::time);
            _take_a_dump();
            fclose(fp);
        }
    }
    return;
}

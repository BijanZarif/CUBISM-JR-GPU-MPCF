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
#include "Histogram.h"
#include <string>
#include <vector>
#include <utility>

class LSRK3_DataMPI
{
public:
    // state of integrator (this implementation is not so nice, should be
    // hidden and an interface base should be used, just sayin')
    static double time;
    static size_t step;

    // accumulation time per process for important branches
    static Histogram histogram;
    static size_t ReportFreq;
    static double t_RHS;    // Convection (GPU)
    static double t_UPDATE; // Update (CPU)
    static double t_COMM;   // MPI communication

    static void notify(double avg_time_rhs, double avg_time_update, double avg_time_comm, double avg_time_BC, const size_t NTIMES)
    {
        histogram.notify("RHS",    (float)avg_time_rhs);
        histogram.notify("UPDATE", (float)avg_time_update);
        histogram.notify("COMM",   (float)avg_time_comm);
        histogram.notify("BC",     (float)avg_time_BC);
        histogram.notify("HALOS",  (float)(avg_time_comm + avg_time_BC));

        if(step % ReportFreq == 0 && step > 0) histogram.consolidate();

        /* if(LSRK3data::step_id % LSRK3data::ReportFreq == 0 && LSRK3data::step_id > 0) */
        /* { */
        /*     double global_t_synch_fs = 0, global_t_bp_fs = 0; */
        /*     double global_avg_time_rhs = 0, global_avg_time_update = 0; */
        /*     double global_t_fs, global_t_up; */

        /*     int global_counter = 0; */

        /*     MPI::COMM_WORLD.Reduce(&t_synch_fs, &global_t_synch_fs, 1, MPI::DOUBLE, MPI::SUM, 0); */
        /*     MPI::COMM_WORLD.Reduce(&t_bp_fs, &global_t_bp_fs, 1, MPI::DOUBLE, MPI::SUM, 0); */
        /*     MPI::COMM_WORLD.Reduce(&counter, &global_counter, 1, MPI::INT, MPI::SUM, 0); */
        /*     MPI::COMM_WORLD.Reduce(&avg_time_rhs, &global_avg_time_rhs, 1, MPI::DOUBLE, MPI::SUM, 0); */
        /*     MPI::COMM_WORLD.Reduce(&avg_time_update, &global_avg_time_update, 1, MPI::DOUBLE, MPI::SUM, 0); */
        /*     MPI::COMM_WORLD.Reduce(&t_fs, &global_t_fs, 1, MPI::DOUBLE, MPI::SUM, 0); */
        /*     MPI::COMM_WORLD.Reduce(&t_up, &global_t_up, 1, MPI::DOUBLE, MPI::SUM, 0); */

        /*     t_synch_fs = t_bp_fs = t_fs = t_up = counter = 0; */

        /*     global_t_synch_fs /= NTIMES; */
        /*     global_t_bp_fs /= NTIMES; */
        /*     global_counter /= NTIMES; */
        /*     global_t_fs /= NTIMES; */
        /*     global_t_up /= NTIMES; */

        /*     const size_t NRANKS = MPI::COMM_WORLD.Get_size(); */

        /*     if (LSRK3data::verbosity >= 1) */
        /*     { */
        /*         cout << "FLOWSTEP: " << avg_time_rhs << " s (per substep), " << avg_time_rhs/NBLOCKS*1e3 << " ms (per block) " << global_avg_time_rhs/NBLOCKS*1e3/NRANKS << " ms (per block per node)" << endl; */

        /*         cout << "TIME LOCALLY AVERAGED FLOWSTEP: " << global_avg_time_rhs/NRANKS <<" s (per substep per node), " << global_avg_time_rhs/NBLOCKS*1e3/NRANKS << " ms (per substep per node per block)" << endl; */

        /*         cout << "TIME GLOBALLY AVERAGED FLOWSTEP: " << global_t_fs/NRANKS/(double)LSRK3data::ReportFreq << " s (per substep)" << endl; */

        /*         cout << "===========================STAGE===========================" << endl; */
        /*         cout << "Synch done in "<< global_counter/NRANKS/(double)LSRK3data::ReportFreq << " passes" << endl; */
        /*         cout << "SYNCHRONIZER FLOWSTEP "<< global_t_synch_fs/NRANKS/(double)LSRK3data::ReportFreq << " s" << endl; */
        /*         cout << "BP FLOWSTEP "<< global_t_bp_fs/NRANKS/(double)LSRK3data::ReportFreq << " s" << endl; */
        /*         cout << "======================================================" << endl; */

        /*         Kflow::printflops(LSRK3data::PEAKPERF_CORE*1e9, LSRK3data::PEAKBAND*1e9, LSRK3data::NCORES, 1,  NBLOCKS*NRANKS, global_t_fs/(double)LSRK3data::ReportFreq/NRANKS); */
        /*         Kupdate::printflops(LSRK3data::PEAKPERF_CORE*1e9, LSRK3data::PEAKBAND*1e9, LSRK3data::NCORES, 1, NBLOCKS*NRANKS, global_t_up/(double)LSRK3data::ReportFreq/NRANKS); */
        /*     } */
        /* } */
    }

};


class LSRK3_IntegratorMPI
{
    const double h, CFL;
    double dt;
    float sos;
    int verbosity;
    std::string SOSkernel, UPkernel;
    const bool isroot;
    bool bhist;

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

        Timer timer;
        double trk1, trk2, trk3;
        vector< pair<double,double> > t_ghosts(3);
        vector< pair<double,double> > t_rhs_up(3);
        {// stage 1
            timer.start();
            t_ghosts[0] = GPU->load_ghosts();
            t_rhs_up[0] = GPU->template process_all<Kupdate>(A1, B1, dtinvh);
            trk1 = timer.stop();
            if (isroot && verbosity) printf("RK stage 1 takes %f sec\n", trk1);
        }
        {// stage 2
            timer.start();
            t_ghosts[1] = GPU->load_ghosts();
            t_rhs_up[1] = GPU->template process_all<Kupdate>(A2, B2, dtinvh);
            trk2 = timer.stop();
            if (isroot && verbosity) printf("RK stage 2 takes %f sec\n", trk2);
        }
        {// stage 3
            timer.start();
            t_ghosts[2] = GPU->load_ghosts();
            t_rhs_up[2] = GPU->template process_all<Kupdate>(A3, B3, dtinvh);
            trk3 = timer.stop();
            if (isroot && verbosity) printf("RK stage 3 takes %f sec\n", trk3);
        }

        const double avg_MPI_comm = (t_ghosts[0].first  + t_ghosts[1].first  + t_ghosts[2].first) / 3.0;
        const double avg_BC_eval  = (t_ghosts[0].second + t_ghosts[1].second + t_ghosts[2].second) / 3.0;
        const double avg_RHS      = (t_rhs_up[0].first  + t_rhs_up[1].first  + t_rhs_up[2].first) / 3.0;
        const double avg_UP       = (t_rhs_up[0].second + t_rhs_up[1].second + t_rhs_up[2].second) / 3.0;

        if (bhist) LSRK3_DataMPI::notify(avg_RHS, avg_UP, avg_MPI_comm, avg_BC_eval, 3);

        return trk1 + trk2 + trk3;
    }

public:

    LSRK3_IntegratorMPI(const GridMPI *grid_, GPUlabMPI *GPU_, const double CFL_, ArgumentParser& parser, const bool isroot_=true) :
        grid(grid_), GPU(GPU_), h(grid_->getH()), CFL(CFL_), isroot(isroot_)
    {
        verbosity = parser("-verb").asInt(0);
        SOSkernel = parser("-SOSkernel").asString("cuda");
        UPkernel  = parser("-UPkernel").asString("cpp");

        // histogram stuff
        bhist = parser("-hist").asBool(false);
        LSRK3_DataMPI::ReportFreq = parser("-histfreq").asInt(100);

        LSRK3_DataMPI::time = 0.0;
        LSRK3_DataMPI::step = 0;
    }

    double operator()(const double dt_max);
};

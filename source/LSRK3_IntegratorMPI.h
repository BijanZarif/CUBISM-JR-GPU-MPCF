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
#include <iostream>
#include <cstdlib>

class LSRK3_DataMPI
{
public:
    // state of integrator (this implementation is not so nice, should be
    // hidden and an interface base should be used, just sayin')
    static double time;
    static size_t step;

    // LSRK3 coefficients
    static Real A1, A2, A3, B1, B2, B3;


    // accumulation time per process for important branches
    static Histogram histogram;
    static size_t ReportFreq;
    static double t_RHS;    // Convection (GPU)
    static double t_UPDATE; // Update (CPU)
    static double t_COMM;   // MPI communication

    static void notify(double avg_time_rhs, double avg_time_update, double avg_time_comm, double avg_time_BC,
            double avg_time_h2d_halo, double avg_time_h2d_in, double avg_time_d2h_out, const size_t NTIMES)
    {
        histogram.notify("RHS",       (float)avg_time_rhs);
        histogram.notify("UPDATE",    (float)avg_time_update);
        histogram.notify("COMM",      (float)avg_time_comm);
        histogram.notify("BC",        (float)avg_time_BC);
        histogram.notify("HALOS",     (float)(avg_time_comm + avg_time_BC));
        histogram.notify("H2D_HALOS", (float)avg_time_h2d_halo);
        histogram.notify("H2D_IN",    (float)avg_time_h2d_in);
        histogram.notify("D2H_OUT",   (float)avg_time_d2h_out);

        if(step % ReportFreq == 0 && step > 0) histogram.consolidate();
    }

};


class LSRK3_IntegratorMPI
{
    const double h;
    double dt, CFL;
    float sos;
    int verbosity;
    std::string SOSkernel, UPkernel, coefficientSet;
    const bool isroot;
    bool bhist;

    const GridMPI* grid;
    GPUlabMPI* GPU;

    template <typename Kupdate>
    double _RKstepGPU(const Real dtinvh)
    {
        Timer timer;
        double trk1, trk2, trk3;
        double t_correction = 0.0;
        vector< pair<double,double> > t_ghosts(3);
        vector< vector<double> > t_main(3);
        {// stage 1
            timer.start();
            t_ghosts[0] = GPU->load_ghosts();
            t_main[0]   = GPU->template process_all<Kupdate>(LSRK3_DataMPI::A1, LSRK3_DataMPI::B1, dtinvh);
            trk1 = timer.stop();
            if (isroot && verbosity) printf("RK stage 1 takes %f sec\n", trk1);
        }
        {// stage 2
            timer.start();
            t_ghosts[1] = GPU->load_ghosts();
            t_main[1]   = GPU->template process_all<Kupdate>(LSRK3_DataMPI::A2, LSRK3_DataMPI::B2, dtinvh);
            trk2 = timer.stop();
            if (isroot && verbosity) printf("RK stage 2 takes %f sec\n", trk2);
        }
        {// stage 3
            timer.start();
            t_ghosts[2] = GPU->load_ghosts();
            t_main[2]   = GPU->template process_all<Kupdate>(LSRK3_DataMPI::A3, LSRK3_DataMPI::B3, dtinvh);
            trk3 = timer.stop();
            if (isroot && verbosity) printf("RK stage 3 takes %f sec\n", trk3);
        }
        {// state correction
            t_correction = GPU->template apply_correction<Kupdate>(-3.0, -6.0);
            if (isroot && verbosity) printf("Correction update takes %f sec\n", t_correction);
        }

        rkstep_timecollector["H2D"] += t_main[0][0] + t_main[1][0] + t_main[2][0] + t_main[0][1] + t_main[1][1] + t_main[2][1];
        rkstep_timecollector["D2H"] += t_main[0][2] + t_main[1][2] + t_main[2][2];
        rkstep_timecollector["RHS"] += t_main[0][3] + t_main[1][3] + t_main[2][3];
        rkstep_timecollector["UP"]  += t_main[0][4] + t_main[1][4] + t_main[2][4];

        if (bhist)
        {
            const double avg_MPI_comm = (t_ghosts[0].first  + t_ghosts[1].first  + t_ghosts[2].first) / 3.0;
            const double avg_BC_eval  = (t_ghosts[0].second + t_ghosts[1].second + t_ghosts[2].second) / 3.0;
            const double avg_h2d_HALO = (t_main[0][0] + t_main[1][0] + t_main[2][0]) / 3.0;
            const double avg_h2d_IN   = (t_main[0][1] + t_main[1][1] + t_main[2][1]) / 3.0;
            const double avg_d2h_OUT  = (t_main[0][2] + t_main[1][2] + t_main[2][2]) / 3.0;
            const double avg_RHS      = (t_main[0][3] + t_main[1][3] + t_main[2][3]) / 3.0;
            const double avg_UP       = (t_main[0][4] + t_main[1][4] + t_main[2][4]) / 3.0;

            LSRK3_DataMPI::notify(avg_RHS, avg_UP, avg_MPI_comm, avg_BC_eval, avg_h2d_HALO, avg_h2d_IN, avg_d2h_OUT, 3);
        }

        return trk1 + trk2 + trk3 + t_correction;
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

        // explicitly set to zero
        LSRK3_DataMPI::time = 0.0;
        LSRK3_DataMPI::step = 0;

        // coefficient set
        coefficientSet = parser("-LSRK3coeffs").asString("gottlieb-shu");
        if (coefficientSet == "gottlieb-shu")
        {
            LSRK3_DataMPI::A1 = 0.0;
            LSRK3_DataMPI::A2 = -2.915492524638791;
            LSRK3_DataMPI::A3 = -0.000000093517376;
            LSRK3_DataMPI::B1 = 0.924574000000000;
            LSRK3_DataMPI::B2 = 0.287713063186749;
            LSRK3_DataMPI::B3 = 0.626538109512740;
        }
        else if (coefficientSet == "williamson")
        {
            LSRK3_DataMPI::A1 = 0.0;
            LSRK3_DataMPI::A2 = -17./32;
            LSRK3_DataMPI::A3 = -32./27;
            LSRK3_DataMPI::B1 = 1./4;
            LSRK3_DataMPI::B2 = 8./9;
            LSRK3_DataMPI::B3 = 3./4;
        }
        else
        {
            std::cerr << "ERROR: Unknown LSRK3 coefficients \"" << coefficientSet << "\"" << std::endl;
            std::exit(-1);
        }
    }

    virtual ~LSRK3_IntegratorMPI()
    {
        grid = 0;
        GPU = 0;
    }

    double operator()(const double dt_max);
    inline void set_CFL(const double LFC) { CFL = LFC; }
};

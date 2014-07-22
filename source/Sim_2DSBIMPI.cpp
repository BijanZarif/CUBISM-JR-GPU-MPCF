/* *
 * Sim_2DSBIMPI.cpp
 *
 * Created by Fabian Wermelinger on 7/19/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "Sim_2DSBIMPI.h"
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <omp.h>
using namespace std;


Sim_2DSBIMPI::Sim_2DSBIMPI(const int argc, const char ** argv, const int isroot) :
    Sim_SteadyStateMPI(argc, argv, isroot)
{
    parser.set_strict_mode();
    const string conf = parser("-config").asString();
    parser.unset_strict_mode();
    if (conf == "xy")
    {
        dims[0] = Coord::X;
        dims[1] = Coord::Y;
        dims[2] = Coord::Z;
    }
    else if (conf == "xz")
    {
        dims[0] = Coord::X;
        dims[2] = Coord::Y;
        dims[1] = Coord::Z;
    }
    else if (conf == "yx")
    {
        dims[1] = Coord::X;
        dims[0] = Coord::Y;
        dims[2] = Coord::Z;
    }
    else if (conf == "yz")
    {
        dims[2] = Coord::X;
        dims[0] = Coord::Y;
        dims[1] = Coord::Z;
    }
    else if (conf == "zx")
    {
        dims[1] = Coord::X;
        dims[2] = Coord::Y;
        dims[0] = Coord::Z;
    }
    else if (conf == "zy")
    {
        dims[2] = Coord::X;
        dims[1] = Coord::Y;
        dims[0] = Coord::Z;
    }
    else
    {
        if (isroot) fprintf(stderr, "ERROR: -config %s is not possible...\n", conf.c_str());
        abort();
    }
}


void Sim_2DSBIMPI::_allocGPU()
{
    if (dims[1] == Coord::X)
    {
        if (isroot) printf("Allocating GPUlab2DSBI_xreflect...\n");
        myGPU = new GPUlab2DSBI_xreflect(*mygrid, nslices, verbosity);
    }
    else if (dims[1] == Coord::Y)
    {
        if (isroot) printf("Allocating GPUlab2DSBI_yreflect...\n");
        myGPU = new GPUlab2DSBI_yreflect(*mygrid, nslices, verbosity);
    }
    else if (dims[1] == Coord::Z)
    {
        if (isroot) printf("Allocating GPUlab2DSBI_zreflect...\n");
        myGPU = new GPUlab2DSBI_zreflect(*mygrid, nslices, verbosity);
    }
}


void Sim_2DSBIMPI::_ic()
{
    if (isroot)
    {
        printf("=====================================================================\n");
        printf("                           2D Shock Bubble                           \n");
        printf("=====================================================================\n");
    }
    const double x0    = parser("-x0").asDouble(0.1);
    const double mach  = parser("-mach").asDouble(1.22);
    const double rho2  = parser("-rho").asDouble(1);
    const double rhob  = parser("-rhobubble").asDouble(0.138);
    const double u2    = parser("-u").asDouble(0.0);
    const double p2    = parser("-p").asDouble(1);
    const double pbub  = parser("-pbubble").asDouble(1);
    const double g1    = parser("-g1").asDouble(1.4);
    const double g2    = parser("-g2").asDouble(1.67);
    const double pc1   = parser("-pc1").asDouble(0.0);
    const double pc2   = parser("-pc2").asDouble(0.0);
    const bool reverse = parser("-reverse").asBool(false);
    SimTools::EPSILON  = (Real)(parser("-mollfactor").asInt(1))*sqrt(3.)*mygrid->getH();

    const Real pre_shock[3] = {rho2, u2, p2};
    Real post_shock[3];
    SimTools::getPostShockRatio(pre_shock, mach, g1, pc1, post_shock);
    const double rho1 = post_shock[0];
    const double u1   = post_shock[1];
    const double p1   = post_shock[2];

    const double G1 = g1-1;
    const double G2 = g2-1;
    const double F1 = g1*pc1;
    const double F2 = g2*pc2;

    vector<Real> Rb; // radius of bubble
    vector<vector<Real> > posb; // position of bubble

    // read bubbles from file
    char fbubble[256];
    if (dims[0]==Coord::X)
        sprintf(fbubble, dims[1]==Coord::Y ? "xybubbles.dat" : "xzbubbles.dat");
    else if (dims[0]==Coord::Y)
        sprintf(fbubble, dims[1]==Coord::X ? "yxbubbles.dat" : "yzbubbles.dat");
    else if (dims[0]==Coord::Z)
        sprintf(fbubble, dims[1]==Coord::X ? "zxbubbles.dat" : "zybubbles.dat");
    ifstream bubbleFile(fbubble);
    if (!bubbleFile.good())
    {
        fprintf(stderr, "Can not load bubble file './%s'. ABORT\n", fname);
        abort();
    }
    uint_t nb;
    bubbleFile >> nb;
    Rb.resize(nb);
    posb.resize(nb);
    for (int i = 0; i < nb; ++i)
    {
        if (!bubbleFile.good())
        {
            fprintf(stderr, "Error reading './%s'. ABORT\n", fname);
            abort();
        }
        posb[i].resize(3);
        bubbleFile >> posb[i][0];
        bubbleFile >> posb[i][1];
        bubbleFile >> posb[i][2];
        bubbleFile >> Rb[i];
    }
    bubbleFile.close();

    // init
    typedef GridMPI::PRIM var;
    const var momentum[3] = {var::U, var::V, var::W};
    const uint_t gridDim[3] = {GridMPI::sizeX, GridMPI::sizeY, GridMPI::sizeZ};
    GridMPI& grid = *mygrid;

#pragma omp paralell for
    for (int iz = 0; iz < gridDim[2]; ++iz)
        for (int iy = 0; iy < gridDim[1]; ++iy)
            for (int ix = 0; ix < gridDim[0]; ++ix)
            {
                double p[3];
                grid.get_pos(ix, iy, iz, p);

                // determine shock region
                const Real shock = SimTools::heaviside(p[dims[0]] - x0);

                // process all bubbles
                Real bubble = 0;;
                for (int i = 0; i < nb; ++i)
                {
                    const Real r = sqrt( pow(p[dims[0]] - posb[i][dims[0]], 2) + pow(p[dims[1]] - posb[i][dims[1]], 2) );
                    bubble = SimTools::heaviside_smooth(r - Rb[i]);
                    if (bubble > 0) break;
                }

                grid(ix, iy, iz, var::R) = shock*rho1 + (1 - shock)*(rhob*bubble + rho2*(1 - bubble));

                grid(ix, iy, iz, momentum[dims[0]]) = (shock*u1 + (1-shock)*u2) * grid(ix, iy, iz, var::R);
                grid(ix, iy, iz, momentum[dims[1]]) = 0;
                grid(ix, iy, iz, momentum[dims[2]]) = 0;

                // phase mix
                grid(ix, iy, iz, var::G) = 1./G1*(1 - bubble) + 1./G2*bubble;
                grid(ix, iy, iz, var::P) = F1/G1*(1 - bubble) + F2/G2*bubble;

                // energy
                const Real pressure  = shock*p1 + (1-shock)*(pbub*bubble + p2*(1 - bubble));
                const Real ke = 0.5*(pow(grid(ix, iy, iz, var::U),2)+pow(grid(ix, iy, iz, var::V),2)+pow(grid(ix, iy, iz, var::W),2))/grid(ix, iy, iz, var::R);
                grid(ix, iy, iz, var::E) = pressure*grid(ix, iy, iz, var::G) + grid(ix, iy, iz, var::P) + ke;
            }
    if (reverse)
    {
        int idx[3];
        int rev[3];
        for (idx[dims[0]]=0; idx[dims[0]]<gridDim[dims[0]]/2; ++idx[dims[0]])
            for (idx[dims[2]]=0; idx[dims[2]]<gridDim[dims[2]]; ++idx[dims[2]])
                for (idx[dims[1]]=0; idx[dims[1]]<gridDim[dims[1]]; ++idx[dims[1]])
                {
                    rev[dims[0]] = gridDim[dims[0]]-1 - idx[dims[0]];
                    rev[dims[1]] = gridDim[dims[1]]-1 - idx[dims[1]];
                    rev[dims[2]] = gridDim[dims[2]]-1 - idx[dims[2]];
                    const Real tmp0 = grid(idx[0],idx[1],idx[2],var::R);
                    const Real tmp1 = -grid(idx[0],idx[1],idx[2],var::U); // 3x minus is ok if the other momenta are zero (which is what they are)
                    const Real tmp2 = -grid(idx[0],idx[1],idx[2],var::V);
                    const Real tmp3 = -grid(idx[0],idx[1],idx[2],var::W);
                    const Real tmp4 = grid(idx[0],idx[1],idx[2],var::E);
                    const Real tmp5 = grid(idx[0],idx[1],idx[2],var::G);
                    const Real tmp6 = grid(idx[0],idx[1],idx[2],var::P);
                    grid(idx[0],idx[1],idx[2],var::R) = grid(rev[0],rev[1],rev[2],var::R);
                    grid(idx[0],idx[1],idx[2],var::U) = -grid(rev[0],rev[1],rev[2],var::U);
                    grid(idx[0],idx[1],idx[2],var::V) = -grid(rev[0],rev[1],rev[2],var::V);
                    grid(idx[0],idx[1],idx[2],var::W) = -grid(rev[0],rev[1],rev[2],var::W);
                    grid(idx[0],idx[1],idx[2],var::E) = grid(rev[0],rev[1],rev[2],var::E);
                    grid(idx[0],idx[1],idx[2],var::G) = grid(rev[0],rev[1],rev[2],var::G);
                    grid(idx[0],idx[1],idx[2],var::P) = grid(rev[0],rev[1],rev[2],var::P);
                    grid(rev[0],rev[1],rev[2],var::R) = tmp0;
                    grid(rev[0],rev[1],rev[2],var::U) = tmp1;
                    grid(rev[0],rev[1],rev[2],var::V) = tmp2;
                    grid(rev[0],rev[1],rev[2],var::W) = tmp3;
                    grid(rev[0],rev[1],rev[2],var::E) = tmp4;
                    grid(rev[0],rev[1],rev[2],var::G) = tmp5;
                    grid(rev[0],rev[1],rev[2],var::P) = tmp6;
                }
    }

    SimTools::symmetry_check<GridMPI>(*mygrid, verbosity, 1.0e-6, dims);
}

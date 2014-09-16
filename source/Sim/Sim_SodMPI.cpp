/* *
 * Sim_SodMPI.cpp
 *
 * Created by Fabian Wermelinger on 7/19/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "Sim_SodMPI.h"
#include <string>
#include <omp.h>
using namespace std;


void Sim_SodMPI::_allocGPU()
{
    if (isroot) printf("Allocating GPUlabMPISod...\n");
    myGPU = new GPUlabMPISod(*mygrid, nslices, verbosity);
}

void Sim_SodMPI::_ic()
{
    if (isroot)
    {
        printf("=====================================================================\n");
        printf("                           Sod Shock Tube                            \n");
        printf("=====================================================================\n");
    }
    const double x0   = parser("-x0").asDouble(0.5);
    const double rho1 = parser("-rho1").asDouble(1.0);
    const double rho2 = parser("-rho2").asDouble(0.125);
    const double u1   = parser("-u1").asDouble(0.0);
    const double u2   = parser("-u2").asDouble(0.0);
    const double p1   = parser("-p1").asDouble(1.0);
    const double p2   = parser("-p2").asDouble(0.1);
    const double g1   = parser("-g1").asDouble(1.4);
    const double g2   = parser("-g2").asDouble(1.4);
    const double pc1  = parser("-pc1").asDouble(0.0);
    const double pc2  = parser("-pc2").asDouble(0.0);
    MaterialDictionary::gamma1 = g1;
    MaterialDictionary::gamma2 = g2;
    MaterialDictionary::pc1 = pc1;
    MaterialDictionary::pc2 = pc2;

    const bool reverse = parser("-reverse").asBool(false);
    const string dir   = parser("-config").asString("x");

    uint_t dims[3] = {Coord::X, Coord::Y, Coord::Z};
    if (dir == "y")
    {
        dims[1] = Coord::X;
        dims[0] = Coord::Y;
        dims[2] = Coord::Z;
    }
    else if (dir == "z")
    {
        dims[1] = Coord::X;
        dims[2] = Coord::Y;
        dims[0] = Coord::Z;
    }

    typedef GridMPI::PRIM var;
    const var momentum[3] = {var::U, var::V, var::W};
    GridMPI& grid = *mygrid;

#pragma omp paralell for
    for (int iz = 0; iz < GridMPI::sizeZ; ++iz)
        for (int iy = 0; iy < GridMPI::sizeY; ++iy)
            for (int ix = 0; ix < GridMPI::sizeX; ++ix)
            {
                double pos[3];
                grid.get_pos(ix, iy, iz, pos);

                // set up along x
                bool x;
                if (reverse)
                    x = pos[dims[0]] > x0;
                else
                    x = pos[dims[0]] < x0;

                const double r = x * rho1 + !x * rho2;
                const double p = x * p1   + !x * p2;
                const double u = x * rho1*u1 + !x * rho2*u2;
                const double G = x * 1./(g1 - 1.) + !x * 1./(g2 - 1.);
                const double P = x * pc1*g1/(g1 - 1.) + !x * pc2*g2/(g2 - 1.);
                assert(r > 0);
                assert(p > 0);
                assert(G > 0);
                assert(P >= 0);

                grid(ix, iy, iz, var::R) = r;
                grid(ix, iy, iz, momentum[dims[0]]) = u;
                grid(ix, iy, iz, momentum[dims[1]]) = 0;
                grid(ix, iy, iz, momentum[dims[2]]) = 0;
                grid(ix, iy, iz, var::E) = G*p + P + 0.5*u*u/r;
                grid(ix, iy, iz, var::G) = G;
                grid(ix, iy, iz, var::P) = P;
            }

    SimTools::symmetry_check<GridMPI>(*mygrid, verbosity, 1.0e-6, dims);
}

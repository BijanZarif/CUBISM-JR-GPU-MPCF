/* *
 * Sim_SICCloudMPI.cpp
 *
 * Created by Fabian Wermelinger on 7/22/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "Sim_SICCloudMPI.h"

#include <cassert>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
using namespace std;

namespace SICCloudData
{
    int n_shapes = 0;
    int n_small = 0;
    int small_count = 0;
    double min_rad = 0;
    double max_rad = 0;
    double seed_s[3], seed_e[3];
    int n_sensors = 0;
    Real nx, ny, nz, Sx, Sy, Sz;
    Real pressureRatio, rho0, c0, u0, p0, rhoB, uB, pB;
    Real rho1, u1, p1, mach;
    Real g1, g2, pc1, pc2;
}


Sim_SICCloudMPI::Sim_SICCloudMPI(const int argc, const char ** argv, const int isroot) :
    Sim_SteadyStateMPI(argc, argv, isroot)
{ }

void Sim_SICCloudMPI::_allocGPU()
{
    if (isroot) printf("Allocating GPUlabSICCloud...\n");
    myGPU = new GPUlabSICCloud(*mygrid, nslices, verbosity);
}

void Sim_SICCloudMPI::_ic()
{
    if (isroot)
    {
        printf("=====================================================================\n");
        printf("                     Shock Induced Cloud Collapse                    \n");
        printf("=====================================================================\n");
    }

    /* parser.set_strict_mode(); */
    // liquid
    SICCloudData::rho0 = parser("-rho0").asDouble(1000);
    SICCloudData::u0   = parser("-u0").asDouble(0);
    SICCloudData::p0   = parser("-p0").asDouble(1);
    // bubbles
    SICCloudData::rhoB = parser("-rhoB").asDouble(1);
    SICCloudData::uB   = parser("-uB").asDouble(0);
    SICCloudData::pB   = parser("-pB").asDouble(0.0234);
    // pressure ratio over shock
    SICCloudData::pressureRatio = parser("-pressureratio").asDouble(40);
    /* parser.unset_strict_mode(); */

    // shock orienation
    SICCloudData::nx = parser("-shockNx").asDouble(1.0);
    SICCloudData::ny = parser("-shockNy").asDouble(0.0);
    SICCloudData::nz = parser("-shockNz").asDouble(0.0);
    // point on shock
    SICCloudData::Sx = parser("-shockSx").asDouble(0.05);
    SICCloudData::Sy = parser("-shockSy").asDouble(0.0);
    SICCloudData::Sz = parser("-shockSz").asDouble(0.0);

    SICCloudData::g1  = parser("-g1").asDouble(6.59);
    SICCloudData::g2  = parser("-g2").asDouble(1.4);
    SICCloudData::pc1 = parser("-pc1").asDouble(4096.0);
    SICCloudData::pc2 = parser("-pc2").asDouble(1.0);

    // normalize shock normal vector
    const Real mag = sqrt(pow(SICCloudData::nx, 2) + pow(SICCloudData::ny, 2) + pow(SICCloudData::nz, 2));
    assert(mag > 0);
    SICCloudData::nx /= mag;
    SICCloudData::ny /= mag;
    SICCloudData::nz /= mag;

    /* *
     * Compute post shock states based on Appendix A of E. Johnsen "Numerical
     * Simulations of Non-Spherical Bubble Collapse", PhD Thesis, Caltech 2007.
     * */
    const Real gamma = SICCloudData::g1;
    const Real pc    = SICCloudData::pc1;
    const Real p0    = SICCloudData::p0;
    const Real rho0  = SICCloudData::rho0;
    const Real u0    = SICCloudData::u0;
    const Real psi   = SICCloudData::pressureRatio;
    const Real p1    = p0*psi;
    const Real tmp1  = (gamma + 1.0)/(gamma - 1.0);
    const Real tmp2  = (p1 + pc)/(p0 + pc);

    SICCloudData::c0   = sqrt(gamma*(p0 + pc)/rho0);
    SICCloudData::mach = sqrt((gamma+1.0)/(2.0*gamma)*(psi - 1.0)*p0/(p0 + pc) + 1.0);
    SICCloudData::u1   = u0 + SICCloudData::c0*(psi - 1.0)*p0/(gamma*(p0 + pc)*SICCloudData::mach);
    SICCloudData::rho1 = rho0*(tmp1*tmp2 + 1.0)/(tmp1 + tmp2);
    SICCloudData::p1   = p1;

    if (verbosity)
    {
        cout << "INITIAL SHOCK" << endl;
        cout << '\t' << "p-Ratio         = " << SICCloudData::pressureRatio << endl;
        cout << '\t' << "Mach            = " << SICCloudData::mach << endl;
        cout << '\t' << "Shock speed     = " << u0 + SICCloudData::c0*SICCloudData::mach<< endl;
        cout << '\t' << "Shock direction = (" << SICCloudData::nx << ", " << SICCloudData::ny << ", " << SICCloudData::nz << ")" << endl;
        cout << '\t' << "Point on shock  = (" << SICCloudData::Sx << ", " << SICCloudData::Sy << ", " << SICCloudData::Sz << ")" << endl << endl;
        cout << "INITIAL CONDITION" << endl;
        cout << '\t' << "Liquid:" << endl;
        cout << "\t\t" << "rho0 = " << SICCloudData::rho0 << endl;
        cout << "\t\t" << "u0   = " << SICCloudData::u0 << endl;
        cout << "\t\t" << "p0   = " << SICCloudData::p0 << endl;
        cout << "\t\t" << "rho1 = " << SICCloudData::rho1 << endl;
        cout << "\t\t" << "u1   = " << SICCloudData::u1 << endl;
        cout << "\t\t" << "p1   = " << SICCloudData::p1 << endl;
        cout << '\t' << "Gas:" << endl;
        cout << "\t\t" << "rhoB = " << SICCloudData::rhoB << endl;
        cout << "\t\t" << "uB   = " << SICCloudData::uB << endl;
        cout << "\t\t" << "pB   = " << SICCloudData::pB << endl << endl;
    }

    Seed<shape> *myseed = NULL;
    _set_cloud(&myseed);
    _ic_quad(myseed);
    delete myseed;
}

void Sim_SICCloudMPI::_initialize_cloud()
{
    ifstream f_read("cloud_config.dat");

    if (!f_read.good())
    {
        cout << "Error: Can not find cloud_config.dat. Abort...\n";
        abort();
    }

    f_read >> SICCloudData::n_shapes >> SICCloudData::n_small;
    f_read >> SICCloudData::min_rad >> SICCloudData::max_rad;
    f_read >> SICCloudData::seed_s[0] >> SICCloudData::seed_s[1] >> SICCloudData::seed_s[2];
    f_read >> SICCloudData::seed_e[0] >> SICCloudData::seed_e[1] >> SICCloudData::seed_e[2];
    if (!f_read.eof())
        f_read >> SICCloudData::n_sensors;
    else
        SICCloudData::n_sensors=0;

    f_read.close();

    if (verbosity)
        printf("cloud data: N %d Nsmall %d Rmin %f Rmax %f s=%f,%f,%f e=%f,%f,%f\n",
                SICCloudData::n_shapes, SICCloudData::n_small, SICCloudData::min_rad, SICCloudData::max_rad,
                SICCloudData::seed_s[0], SICCloudData::seed_s[1], SICCloudData::seed_s[2],
                SICCloudData::seed_e[0], SICCloudData::seed_e[1], SICCloudData::seed_e[2]);
}

void Sim_SICCloudMPI::_set_cloud(Seed<shape> **seed)
{
    const MPI_Comm cart_world = mygrid->getCartComm();

    if(isroot) _initialize_cloud(); // requires file cloud_config.dat

    MPI_Bcast(&SICCloudData::n_shapes,    1, MPI::INT,    0, cart_world);
    MPI_Bcast(&SICCloudData::n_small,     1, MPI::INT,    0, cart_world);
    MPI_Bcast(&SICCloudData::small_count, 1, MPI::INT,    0, cart_world);
    MPI_Bcast(&SICCloudData::min_rad,     1, MPI::DOUBLE, 0, cart_world);
    MPI_Bcast(&SICCloudData::max_rad,     1, MPI::DOUBLE, 0, cart_world);
    MPI_Bcast(SICCloudData::seed_s,       3, MPI::DOUBLE, 0, cart_world);
    MPI_Bcast(SICCloudData::seed_e,       3, MPI::DOUBLE, 0, cart_world);
    MPI_Bcast(&SICCloudData::n_sensors,   1, MPI::INT,    0, cart_world);

    Seed<shape> * const newseed = new Seed<shape>(SICCloudData::seed_s, SICCloudData::seed_e);
    assert(newseed != NULL);

    vector<shape> v(SICCloudData::n_shapes);

    if(isroot)
    {
        newseed->make_shapes(SICCloudData::n_shapes, "cloud.dat", mygrid->getH());
        v = newseed->get_shapes();
    }

    MPI_Bcast(&v.front(), v.size() * sizeof(shape), MPI::CHAR, 0, cart_world);

    if (!isroot) newseed->set_shapes(v);

    assert(newseed->get_shapes_size() > 0 && newseed->get_shapes_size() == SICCloudData::n_shapes);

    double myorigin[3], myextent[3];
    mygrid->get_origin(myorigin);
    mygrid->get_extent(myextent);
    *newseed = newseed->retain_shapes(myorigin, myextent);

    *seed = newseed;

    MPI_Barrier(cart_world);
}

static Real is_shock(const Real P[3])
{
    const Real nx = SICCloudData::nx;
    const Real ny = SICCloudData::ny;
    const Real nz = SICCloudData::nz;
    const Real Sx = SICCloudData::Sx;
    const Real Sy = SICCloudData::Sy;
    const Real Sz = SICCloudData::Sz;

    // no need to divide by n*n, since n*n = 1
    const Real d = -(nx*(Sx-P[0]) + ny*(Sy-P[1]) + nz*(Sz-P[2]));

    return SimTools::heaviside(d);
    /* return Simulation_Environment::heaviside_smooth(d); */
}

struct FluidElement
{
    Real rho, u, v, w, energy, G, P;
    FluidElement() : rho(-1), u(0), v(0), w(0), energy(-1), G(-1), P(-1) { }
    FluidElement(const FluidElement& e): rho(e.rho), u(e.u), v(e.v), w(e.w), energy(e.energy), G(e.G), P(e.P) { }
    FluidElement& operator=(const FluidElement& e)
    {
        if (this != &e) {rho = e.rho; u = e.u; v = e.v; w = e.w; energy = e.energy; G = e.G; P = e.P;}
        return *this;
    }
    FluidElement operator+(const FluidElement& e) const
    {
        FluidElement sum(e);
        sum.rho += rho; sum.u += u; sum.v += v; sum.w += w; sum.energy += energy; sum.G += G; sum.P += P;
        return sum;
    }
    friend FluidElement operator*(const Real s, const FluidElement& e);
};

FluidElement operator*(const Real s, const FluidElement& e)
{
    FluidElement prod(e);
    prod.rho *= s; prod.u *= s; prod.v *= s; prod.w *= s; prod.energy *= s; prod.G *= s; prod.P *= s;
    return prod;
}

template <typename T>
static T set_IC(const Real shock, const Real bubble)
{
    T out;

    const Real pre_shock[3]    = {SICCloudData::rho0, SICCloudData::u0, SICCloudData::p0};
    const Real post_shock[3]   = {SICCloudData::rho1, SICCloudData::u1, SICCloudData::p1};
    const Real bubble_state[3] = {SICCloudData::rhoB, SICCloudData::uB, SICCloudData::pB};

    const Real G1 = SICCloudData::g1-1;
    const Real G2 = SICCloudData::g2-1;
    const Real F1 = SICCloudData::g1*SICCloudData::pc1;
    const Real F2 = SICCloudData::g2*SICCloudData::pc2;

    out.rho = shock*post_shock[0] + (1-shock)*(bubble_state[0]*bubble + pre_shock[0]*(1-bubble));
    out.u   = (shock*post_shock[1] + (1-shock)*(bubble_state[1]*bubble + pre_shock[1]*(1-bubble)))*out.rho;
    out.v   = 0.0;
    out.w   = 0.0;

    // component mix (Note: shock = 1 implies bubble = 0)
    out.G = 1./G1*(1-bubble) + 1./G2*bubble;
    out.P = F1/G1*(1-bubble) + F2/G2*bubble;

    // energy
    const Real pressure  = shock*post_shock[2] + (1-shock)*(bubble_state[2]*bubble + pre_shock[2]*(1-bubble));
    const Real ke = 0.5*(out.u*out.u + out.v*out.v + out.w*out.w)/out.rho;
    out.energy = pressure*out.G + out.P + ke;

    return out;
}


template <typename T>
static T get_IC(const Real pos[3], const vector<shape> * const blockShapes)
{
    T IC;

    const Real shock = is_shock(pos);

    /* *
     * For the IC, bubbles are not allowed to be located in the post shock
     * region.  No check is being performed.
     * */
    if (!blockShapes)
    {
        // If blockShapes == NULL, it is assumed there are no shapes in this
        // block.  Therefore, only the shock wave is treated for initial
        // conditions.
        IC = set_IC<T>(shock, 0);
    }
    else
    {
        // There are shapes within this block to be taken care of.
        const Real bubble = eval(*blockShapes, pos);
        IC = set_IC<T>(shock, bubble);
    }

    return IC;
}

template<typename T>
static T integral(const double p[3], const Real h, const vector<shape> * const blockShapes) // h should be cubism h/2
{
    T samples[3][3][3];
    T zintegrals[3][3];
    T yzintegrals[3];

    const Real x0[3] = {p[0] - h, p[1] - h, p[2] - h};

    for(int iz=0; iz<3; iz++)
        for(int iy=0; iy<3; iy++)
            for(int ix=0; ix<3; ix++)
            {
                const Real mypos[3] = {x0[0]+ix*h, x0[1]+iy*h, x0[2]+iz*h};
                samples[iz][iy][ix] = get_IC<T>(mypos, blockShapes);
            }

    for(int iy=0; iy<3; iy++)
        for(int ix=0; ix<3; ix++)
            zintegrals[iy][ix] = (1/6.) * samples[0][iy][ix]+(2./3) * samples[1][iy][ix]+(1/6.)* samples[2][iy][ix];

    for(int ix=0; ix<3; ix++)
        yzintegrals[ix] = (1./6)*zintegrals[0][ix] + (2./3)*zintegrals[1][ix]+(1./6)*zintegrals[2][ix];

    return (1./6) * yzintegrals[0]+(2./3) * yzintegrals[1]+(1./6)* yzintegrals[2];
}

void Sim_SICCloudMPI::_ic_quad(const Seed<shape> * const seed)
{
    GridMPI& grid = *mygrid;
    const Real h  = grid.getH();
    vector<shape> v_shapes = seed->get_shapes();

    typedef GridMPI::PRIM var;
#pragma omp parallel for
    for (int iz = 0; iz < GridMPI::sizeZ; ++iz)
        for (int iy = 0; iy < GridMPI::sizeY; ++iy)
            for (int ix = 0; ix < GridMPI::sizeX; ++ix)
            {
                double p[3];
                grid.get_pos(ix, iy, iz, p);
                FluidElement IC = integral<FluidElement>(p, h/2, &v_shapes);

                grid(ix, iy, iz, var::R) = IC.rho;
                grid(ix, iy, iz, var::U) = IC.u;
                grid(ix, iy, iz, var::V) = IC.v;
                grid(ix, iy, iz, var::W) = IC.w;
                grid(ix, iy, iz, var::E) = IC.energy;
                grid(ix, iy, iz, var::G) = IC.G;
                grid(ix, iy, iz, var::P) = IC.P;
            }
}

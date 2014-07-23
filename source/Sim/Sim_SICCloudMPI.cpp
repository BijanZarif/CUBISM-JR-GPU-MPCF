/* *
 * Sim_SICCloudMPI.cpp
 *
 * Created by Fabian Wermelinger on 7/22/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "Sim_SICCloudMPI.h"

#include <fstream>
#include <vector>
#include <iostream>
#include <cstdlib>
using namespace std;

namespace CloudData
{
    int n_shapes = 0;
    int n_small = 0;
    int small_count = 0;
    double min_rad = 0;
    double max_rad = 0;
    double seed_s[3], seed_e[3];
    int n_sensors = 0;
}


Sim_SICCloudMPI::Sim_SICCloudMPI(const int argc, const char ** argv, const int isroot) :
    Sim_SteadyStateMPI(argc, argv, isroot), cart_world(mygrid->getCartComm())
{
}

void Sim_SICCloudMPI::_allocGPU()
{
    if (isroot) printf("Allocating GPUlabSICCloud...\n");
    myGPU = new GPUlabSICCloud(*mygrid, nslices, verbosity);
}

void Sim_SICCloudMPI::_ic()
{
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

    f_read >> CloudData::n_shapes >> CloudData::n_small;
    f_read >> CloudData::min_rad >> CloudData::max_rad;
    f_read >> CloudData::seed_s[0] >> CloudData::seed_s[1] >> CloudData::seed_s[2];
    f_read >> CloudData::seed_e[0] >> CloudData::seed_e[1] >> CloudData::seed_e[2];
    if (!f_read.eof())
        f_read >> CloudData::n_sensors;
    else
        CloudData::n_sensors=0;

    f_read.close();

    if (verbosity)
        printf("cloud data: N %d Nsmall %d Rmin %f Rmax %f s=%f,%f,%f e=%f,%f,%f\n",
                CloudData::n_shapes, CloudData::n_small, CloudData::min_rad, CloudData::max_rad,
                CloudData::seed_s[0], CloudData::seed_s[1], CloudData::seed_s[2],
                CloudData::seed_e[0], CloudData::seed_e[1], CloudData::seed_e[2]);
}

void Sim_SICCloudMPI::_set_cloud(Seed<shape> **seed)
{
    if(isroot) _initialize_cloud(); // requires file cloud_config.dat

    MPI_Bcast(&CloudData::n_shapes,    1, MPI::INT,    0, cart_world);
    MPI_Bcast(&CloudData::n_small,     1, MPI::INT,    0, cart_world);
    MPI_Bcast(&CloudData::small_count, 1, MPI::INT,    0, cart_world);
    MPI_Bcast(&CloudData::min_rad,     1, MPI::DOUBLE, 0, cart_world);
    MPI_Bcast(&CloudData::max_rad,     1, MPI::DOUBLE, 0, cart_world);
    MPI_Bcast(CloudData::seed_s,       3, MPI::DOUBLE, 0, cart_world);
    MPI_Bcast(CloudData::seed_e,       3, MPI::DOUBLE, 0, cart_world);
    MPI_Bcast(&CloudData::n_sensors,   1, MPI::INT,    0, cart_world);

    Seed<shape> * const newseed = new Seed<shape>(CloudData::seed_s, CloudData::seed_e);
    assert(newseed != NULL);

    vector<shape> v(CloudData::n_shapes);

    if(isroot)
    {
        newseed->make_shapes(CloudData::n_shapes, "cloud.dat", mygrid->getH());
        v = newseed->get_shapes();
    }

    MPI_Bcast(&v.front(), v.size() * sizeof(shape), MPI::CHAR, 0, cart_world);

    if (!isroot) newseed->set_shapes(v);

    assert(newseed->get_shapes_size() > 0 && newseed->get_shapes_size() == CloudData::n_shapes);

    double myorigin[3], myextent[3];
    mygrid->get_origin(myorigin);
    mygrid->get_extent(myextent);
    *newseed = newseed->retain_shapes(myorigin, myextent);

    *seed = newseed;

    MPI_Barrier(cart_world);
}

void Sim_SICCloudMPI::_ic_quad(const Seed<shape> * const seed)
{
}

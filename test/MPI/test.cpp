/* *
 * test.cpp
 *
 * Created by Fabian Wermelinger on 6/18/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
using namespace std;

#include "ArgumentParser.h"
#include "GridMPI.h"
#include "Timer.h"
#include "HDF5Dumper_MPI.h"
#include "SerializerIO_WaveletCompression_MPI_Simple.h"

#ifndef _BLOCKSIZE_
#define _BLOCKSIZE_ 16
#endif


struct StreamerGridPointIterative //dummy
{
    static const int channels = 1;

    const std::vector<Real *>& soa_data;
    static const int NX = GridMPI::sizeX;
    static const int NY = GridMPI::sizeY;
    static const int NZ = GridMPI::sizeZ;

    StreamerGridPointIterative() : soa_data(std::vector<Real *>(7,NULL)) {} // can not access data!
    StreamerGridPointIterative(const std::vector<Real *>& pdata) : soa_data(pdata) {}

    template<int channel>
    inline Real operate(const int ix, const int iy, const int iz) { abort(); return 0; }

    const char * name() { return "StreamerGridPointIterative" ; }
};


template<> inline Real StreamerGridPointIterative::operate<0>(const int ix, const int iy, const int iz) { const int idx = ix + NX * (iy + NY * iz); return soa_data[0][idx]; }
/* template<> inline Real StreamerGridPointIterative::operate<1>(const FluidElement& e) { return e.u/e.rho; } */
/* template<> inline Real StreamerGridPointIterative::operate<2>(const FluidElement& e) { return e.v/e.rho; } */
/* template<> inline Real StreamerGridPointIterative::operate<3>(const FluidElement& e) { return e.w/e.rho; } */
/* template<> inline Real StreamerGridPointIterative::operate<4>(const FluidElement& e) { return (e.energy-0.5*(e.u*e.u+e.v*e.v+e.w*e.w)/e.rho - e.P)/e.G; } */
/* template<> inline Real StreamerGridPointIterative::operate<5>(const FluidElement& e) { return e.G; } */
/* template<> inline Real StreamerGridPointIterative::operate<6>(const FluidElement& e) { return e.P; } */
/* template<> inline Real StreamerGridPointIterative::operate<7>(const FluidElement& e) { return e.dummy; } */


template <typename TGrid>
struct myStreamer
{
    static const int NCHANNELS = 9;

    const std::vector<Real *>& soa_data;
    const int NX = TGrid::sizeX;
    const int NY = TGrid::sizeY;
    const int NZ = TGrid::sizeZ;

    myStreamer(const TGrid& grid) : soa_data(grid.pdata()) {}

    void operate(const int ix, const int iy, const int iz, Real out[NCHANNELS]) const
    {
        const int idx = ix + NX * (iy + NY * iz);
        assert(idx < NX * NY * NZ);
        for (int i = 0; i < 7; ++i)
            out[i] = soa_data[i][idx];
    }

    void operate(const Real input[NCHANNELS], const int ix, const int iy, const int iz) const
    {
        const int idx = ix + NX * (iy + NY * iz);
        assert(idx < NX * NY * NZ);
        const Real * const pRHO = soa_data[0];
        pRHO[idx] = input[0];
    }

    static const char * getAttributeName() { return "Tensor"; }
    /* static const char * getAttributeName() { return "Scalar"; } */
};



int main(int argc, const char *argv[])
{
    MPI_Init(&argc, const_cast<char***>(&argv));

    /* int provided; */
    /* MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); */

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const bool isroot = world_rank == 0;

#if 0
    if (isroot) printf("Comm world_size is %d\n", world_size);

    int thread_support;
    MPI_Query_thread(&thread_support);
    if (isroot) printf("Thread support is %d\n", thread_support);
    if (isroot) printf("Thread support (init) is %d\n", provided);
    if (isroot) printf("MPI_THREAD_MULTIPLE = %d\n", MPI_THREAD_MULTIPLE);

#pragma omp parallel
    {
        int ismain = 0;
        MPI_Is_thread_main(&ismain);

        const int tid = omp_get_thread_num();
        const int Nth = omp_get_num_threads();

        printf(ismain? "Hi from thread %d (i am main) on rank %d\n" : "Hi from thread %d on rank %d\n", tid, world_rank);

#endif

    // create cartesian topology
    ArgumentParser parser(argc, argv);
    const int npex = parser("-npex").asInt(2);
    const int npey = parser("-npey").asInt(2);
    const int npez = parser("-npez").asInt(2);

    Timer t1;
    t1.start();
    GridMPI grid(npex, npey, npez);
    const double tG = t1.stop();
    /* printf("Rank %d init GridMPI %f\n", world_rank, tG); */

    t1.start();
    grid.send_receive_all();
    const double tS = t1.stop();
    /* printf("Rank %d init send-receive %f\n", world_rank, tS); */

#if 1
    /* //funky test */
    /* Real *prho = grid.pdata()[0]; */
    /* for (int i = 0; i < grid.size(); ++i) */
    /*     prho[i] = world_rank; */
    /*     /1* prho[i] = world_rank*grid.size() + i; *1/ */

    // groovy test
    double pos[3];
    typedef NodeBlock::PRIM var;
    for (int iz = 0; iz < GridMPI::sizeZ; ++iz)
        for (int iy = 0; iy < GridMPI::sizeY; ++iy)
            for (int ix = 0; ix < GridMPI::sizeX; ++ix)
            {
                const unsigned int idx = ix + GridMPI::sizeX * (iy + GridMPI::sizeY * iz);
                grid.get_pos(ix, iy, iz, pos);

                Real val;
                if (pos[0] < 0.5)
                    val = 1.0;
                else
                    val = 0.0;

                grid(ix, iy, iz, var::R) = val;
                /* grid(ix, iy, iz, var::R) = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]); */
                /* grid(ix, iy, iz, var::U) = pos[0]; */
                /* grid(ix, iy, iz, var::V) = pos[1]; */
                /* grid(ix, iy, iz, var::W) = pos[2]; */
            }
#endif

    SerializerIO_WaveletCompression_MPI_SimpleBlocking<GridMPI, StreamerGridPointIterative> mywaveletdumper;
    mywaveletdumper.verbose();
    mywaveletdumper.set_threshold(5e-2);
    mywaveletdumper.Write<0>(grid, "test");

    /* DumpHDF5_MPI<GridMPI, myStreamer<GridMPI> >(grid, 0, "test"); */


#if 0
    printf("Rank %i coords (%i,%i,%i)\n", cart_rank, coords[0], coords[1], coords[2]);
#endif

    // test MPI_Allreduce
    /* int maxRank; */
    /* MPI_Allreduce(&myrank, &maxRank, 1, MPI_INT, MPI_MAX, cart_world); */
    /* printf("Rank %d received maxRank %d\n", myrank, maxRank); */

    MPI_Finalize();

    return 0;
}

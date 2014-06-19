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
using namespace std;

#include "ArgumentParser.h"
#include "GridMPI.h"
#include "Timer.h"

#ifndef _BLOCKSIZE_
#define _BLOCKSIZE_ 16
#endif


// global cartesian communicator


// main data
/* template <typename T> */
/* struct Data */
/* { */
/*     const unsigned int N, Nx, Ny, Nz; */
/*     vector<T> data; */
/*     vector<T> xghost_l, xghost_r; */
/*     vector<T> yghost_l, yghost_r; */
/*     vector<T> zghost_l, zghost_r; */

/*     Data(const unsigned int _Nx, const unsigned int _Ny, const unsigned int _Nz) */
/*         : */
/*             N(_Nx * _Ny * _Nz), Nx(_Nx), Ny(_Ny), Nz(_Nz), */
/*             data(N, -1), */
/*             xghost_l(3 * _Ny * _Nz, -1), xghost_r(3 * _Ny * _Nz, -1), */
/*             yghost_l(_Nx * 3 * _Nz, -1), yghost_r(_Nx * 3 * _Nz, -1), */
/*             zghost_l(_Nx * _Ny * 3, -1), zghost_r(_Nx * _Ny * 3, -1) */
/*     { } */

/*     T& operator()(const int ix, const int iy, const int iz) */
/*     { */
/*         return data[ix + Nx * (iy + Ny * iz)]; */
/*     } */
/* }; */


#if 0
void get_nbr_ranks(int nbr_ranks[6], const int mycoords[3], const int dims[3])
{
    /* *
     * Returns ranks of the neighbor processes in the (periodic) 3D cartesian
     * topology in nbr_ranks = {xrank_l, xrank_r, yrank_l, yrank_r, zrank_l,
     * zrank_r}
     * */
    int coords_l[3], coords_r[3];
    for (int i = 0; i < 3; ++i)
    {
        const int im1 = (i-1 + 3) % 3;
        const int ip1 = (i+1 + 3) % 3;
        coords_l[i]   = (mycoords[i]-1 + dims[i]) % dims[i];
        coords_l[im1] = mycoords[im1];
        coords_l[ip1] = mycoords[ip1];
        coords_r[i]   = (mycoords[i]+1 + dims[i]) % dims[i];
        coords_r[im1] = mycoords[im1];
        coords_r[ip1] = mycoords[ip1];
        MPI_Cart_rank(cart_world, coords_l, &nbr_ranks[i*2 + 0]);
        MPI_Cart_rank(cart_world, coords_r, &nbr_ranks[i*2 + 1]);
    }
}


// send and receive
template <typename T>
void send_receive(T *ghost_l, T *ghost_r, const int Ng, const int myrank, const int left, const int right)
{
    vector<T> recv_buffer_l(Ng);
    vector<T> recv_buffer_r(Ng);

    MPI_Status status;
    MPI_Sendrecv(ghost_l, Ng, _MPI_DATA_TYPE_, left,  1, &recv_buffer_r[0], Ng, _MPI_DATA_TYPE_, right, MPI_ANY_TAG, cart_world, &status);
    MPI_Sendrecv(ghost_r, Ng, _MPI_DATA_TYPE_, right, 2, &recv_buffer_l[0], Ng, _MPI_DATA_TYPE_, left,  MPI_ANY_TAG, cart_world, &status);

    //copy back
    for (int i = 0; i < Ng; ++i)
    {
        ghost_l[i] = recv_buffer_l[i];
        ghost_r[i] = recv_buffer_r[i];
    }
}

template <typename T>
void send_receive_all(NodeBlock& myblock, const int myrank, const int mycoords[3], const int dims[3])
{
    int nbr[6];
    get_nbr_ranks(nbr, mycoords, dims);

    const int Ng = myblock.size_ghost();

    send_receive<T>(myblock.pxghost_l()[0], myblock.pxghost_r()[0], Ng, myrank, nbr[0], nbr[1]);
    send_receive<T>(myblock.pyghost_l()[0], myblock.pyghost_r()[0], Ng, myrank, nbr[2], nbr[3]);
    send_receive<T>(myblock.pzghost_l()[0], myblock.pzghost_r()[0], Ng, myrank, nbr[4], nbr[5]);
}
#endif


// utils
/* template <typename T> */
/* void dump(Data<T>& mydata, const int rank) */
/* { */
/*     ostringstream fname; */
/*     fname << "rank_" << rank; */
/*     ofstream out(fname.str().c_str()); */

/*     for (int iz = 0; iz < mydata.Nz; ++iz) */
/*     { */
/*         for (int iy = 0; iy < mydata.Ny; ++iy) */
/*         { */
/*             for (int ix = 0; ix < mydata.Nx; ++ix) */
/*                 out << mydata(ix,iy,iz) << '\t'; */
/*             out << endl; */
/*         } */
/*         out << endl; */
/*     } */
/*     out.close(); */
/* } */


/* template <typename T> */
/* void dump_ghosts(Data<T>& mydata, const int rank) */
/* { */
/*     ostringstream fname; */
/*     fname << "ghosts_rank_" << rank; */
/*     ofstream out(fname.str().c_str()); */

/*     out << "XGHOSTS:" << endl; */
/*     out << "===LEFT===" << endl; */
/*     for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i) */
/*         out << mydata.xghost_l[i] << '\t'; */
/*     out << endl; */
/*     out << "===RIGHT===" << endl; */
/*     for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i) */
/*         out << mydata.xghost_r[i] << '\t'; */
/*     out << endl; */

/*     out << endl; */

/*     out << "YGHOSTS:" << endl; */
/*     out << "===LEFT===" << endl; */
/*     for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i) */
/*         out << mydata.yghost_l[i] << '\t'; */
/*     out << endl; */
/*     out << "===RIGHT===" << endl; */
/*     for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i) */
/*         out << mydata.yghost_r[i] << '\t'; */
/*     out << endl; */

/*     out << endl; */

/*     out << "ZGHOSTS:" << endl; */
/*     out << "===LEFT===" << endl; */
/*     for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i) */
/*         out << mydata.zghost_l[i] << '\t'; */
/*     out << endl; */
/*     out << "===RIGHT===" << endl; */
/*     for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i) */
/*         out << mydata.zghost_r[i] << '\t'; */
/*     out << endl; */

/*     out.close(); */
/* } */




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
    /* GridMPI<NodeBlock> grid(npex, npey, npez); */


    /* assert(nx_proc * ny_proc * nz_proc == world_size); */

    /* int dims[3]  = {nx_proc, ny_proc, nz_proc}; */
    /* int periodic[3] = {1, 1, 1}; */
    /* MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periodic, true, &cart_world); */

    /* int mysize, myrank; */
    /* int mycoords[3]; */
    /* MPI_Comm_size(cart_world, &mysize); */
    /* MPI_Comm_rank(cart_world, &myrank); */
    /* MPI_Cart_coords(cart_world, myrank, 3, mycoords); */

#if 0
    printf("Rank %i coords (%i,%i,%i)\n", cart_rank, coords[0], coords[1], coords[2]);
#endif

    // init block data for local process
    /* Data<data_t> mydata(_BLOCKSIZE_, _BLOCKSIZE_, _BLOCKSIZE_); */
    /* for (int i = 0; i < mydata.N; ++i) */
    /*     mydata.data[i] = myrank; */
    /* for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i) */
    /* { */
    /*     mydata.xghost_l[i] = mydata.xghost_r[i] = myrank; */
    /*     mydata.yghost_l[i] = mydata.yghost_r[i] = myrank; */
    /*     mydata.zghost_l[i] = mydata.zghost_r[i] = myrank; */
    /* } */
    /* NodeBlock myblock; */
    /* myblock.clear(); */


    // send/receive ghosts
    /* send_receive_all<Real>(myblock, myrank, mycoords, dims); */


    // test MPI_Allreduce
    /* int maxRank; */
    /* MPI_Allreduce(&myrank, &maxRank, 1, MPI_INT, MPI_MAX, cart_world); */
    /* printf("Rank %d received maxRank %d\n", myrank, maxRank); */



    /* // dump */
    /* dump<data_t>(mydata, myrank); */
    /* dump_ghosts<data_t>(mydata, myrank); */


    MPI_Finalize();

    return 0;
}

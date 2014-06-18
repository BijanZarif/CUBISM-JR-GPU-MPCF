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

#ifndef _BLOCKSIZE_
#define _BLOCKSIZE_ 16
#endif


// global cartesian communicator
MPI_Comm cart_world;


// main data
template <typename T>
struct Data
{
    const unsigned int N, Nx, Ny, Nz;
    vector<T> data;
    vector<T> xghost_l, xghost_r;
    vector<T> yghost_l, yghost_r;
    vector<T> zghost_l, zghost_r;

    Data(const unsigned int _Nx, const unsigned int _Ny, const unsigned int _Nz)
        :
            N(_Nx * _Ny * _Nz), Nx(_Nx), Ny(_Ny), Nz(_Nz),
            data(N, -1),
            xghost_l(3 * _Ny * _Nz, -1), xghost_r(3 * _Ny * _Nz, -1),
            yghost_l(_Nx * 3 * _Nz, -1), yghost_r(_Nx * 3 * _Nz, -1),
            zghost_l(_Nx * _Ny * 3, -1), zghost_r(_Nx * _Ny * 3, -1)
    { }

    T& operator()(const int ix, const int iy, const int iz)
    {
        return data[ix + Nx * (iy + Ny * iz)];
    }
};


// send and receive
template <typename T>
void x_send_receive(vector<T>& ghost_l, vector<T>& ghost_r, const int myrank, const int mycoords[3], const int Nx)
{
    vector<T> recv_buffer_l(ghost_l.size());
    vector<T> recv_buffer_r(ghost_r.size());

    const int xi_left  = (mycoords[0]-1 + Nx) % Nx;
    const int xi_right = (mycoords[0]+1 + Nx) % Nx;
    int coords_l[3] = {xi_left,  mycoords[1], mycoords[2]};
    int coords_r[3] = {xi_right, mycoords[1], mycoords[2]};
    int rank_l, rank_r;
    MPI_Cart_rank(cart_world, coords_l, &rank_l);
    MPI_Cart_rank(cart_world, coords_r, &rank_r);

    MPI_Status status;
    MPI_Sendrecv(&ghost_l[0], ghost_l.size(), MPI_INT, rank_l, 1, &recv_buffer_r[0], recv_buffer_r.size(), MPI_INT, rank_r, MPI_ANY_TAG, cart_world, &status);
    MPI_Sendrecv(&ghost_r[0], ghost_r.size(), MPI_INT, rank_r, 3, &recv_buffer_l[0], recv_buffer_l.size(), MPI_INT, rank_l, MPI_ANY_TAG, cart_world, &status);

    ghost_l = recv_buffer_l;
    ghost_r = recv_buffer_r;
}

template <typename T>
void send_receive_all(Data<T>& mydata, const int myrank, const int mycoords[3], const int mpi_grid_dim[3])
{
    x_send_receive<T>(mydata.xghost_l, mydata.xghost_r, myrank, mycoords, mpi_grid_dim[0]);
}


// utils
template <typename T>
void dump(Data<T>& mydata, const int rank)
{
    ostringstream fname;
    fname << "rank_" << rank;
    ofstream out(fname.str().c_str());

    for (int iz = 0; iz < mydata.Nz; ++iz)
    {
        for (int iy = 0; iy < mydata.Ny; ++iy)
        {
            for (int ix = 0; ix < mydata.Nx; ++ix)
                out << mydata(ix,iy,iz) << '\t';
            out << endl;
        }
        out << endl;
    }
    out.close();
}


template <typename T>
void dump_ghosts(Data<T>& mydata, const int rank)
{
    ostringstream fname;
    fname << "ghosts_rank_" << rank;
    ofstream out(fname.str().c_str());

    out << "XGHOSTS:" << endl;
    out << "===LEFT===" << endl;
    for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i)
        out << mydata.xghost_l[i] << '\t';
    out << endl;
    out << "===RIGHT===" << endl;
    for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i)
        out << mydata.xghost_r[i] << '\t';
    out << endl;

    out << endl;

    out << "YGHOSTS:" << endl;
    out << "===LEFT===" << endl;
    for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i)
        out << mydata.yghost_l[i] << '\t';
    out << endl;
    out << "===RIGHT===" << endl;
    for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i)
        out << mydata.yghost_r[i] << '\t';
    out << endl;

    out << endl;

    out << "ZGHOSTS:" << endl;
    out << "===LEFT===" << endl;
    for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i)
        out << mydata.zghost_l[i] << '\t';
    out << endl;
    out << "===RIGHT===" << endl;
    for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i)
        out << mydata.zghost_r[i] << '\t';
    out << endl;

    out.close();
}




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
    const int nx_proc = parser("-nxproc").asInt(2);
    const int ny_proc = parser("-nyproc").asInt(2);
    const int nz_proc = parser("-nzproc").asInt(2);
    assert(nx_proc * ny_proc * nz_proc == world_size);

    int dims[3]  = {nx_proc, ny_proc, nz_proc};
    int periodic[3] = {1, 1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periodic, true, &cart_world);

    int cart_size, cart_rank;
    int mpi_coords[3];
    MPI_Comm_size(cart_world, &cart_size);
    MPI_Comm_rank(cart_world, &cart_rank);
    MPI_Cart_coords(cart_world, cart_rank, 3, mpi_coords);

#if 0
    printf("Rank %i coords (%i,%i,%i)\n", cart_rank, coords[0], coords[1], coords[2]);
#endif

    // init block data for local process
    typedef int data_t;
    Data<data_t> mydata(_BLOCKSIZE_, _BLOCKSIZE_, _BLOCKSIZE_);
    for (int i = 0; i < mydata.N; ++i)
        mydata.data[i] = cart_rank;
    for (int i = 0; i < 3*_BLOCKSIZE_*_BLOCKSIZE_; ++i)
    {
        mydata.xghost_l[i] = mydata.xghost_r[i] = cart_rank;
        mydata.yghost_l[i] = mydata.yghost_r[i] = cart_rank;
        mydata.zghost_l[i] = mydata.zghost_r[i] = cart_rank;
    }

    // send/receive ghosts
    send_receive_all<data_t>(mydata, cart_rank, mpi_coords, dims);




    /* // dump */
    /* dump<data_t>(mydata, cart_rank); */
    dump_ghosts<data_t>(mydata, cart_rank);


    MPI_Finalize();

    return 0;
}

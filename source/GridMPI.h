/*
 *  GridMPI.h
 *  FacesMPI
 *
 *  Created by Diego Rossinelli on 10/21/11.
 *  Modified for SOA/GPU representation by Fabian Wermelinger on 6/19/14
 *  Copyright 2011/14 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <mpi.h>
#include <vector>
#include <cmath>
#include "NodeBlock.h"

using namespace std;

#ifdef _FLOAT_PRECISION_
#define _MPI_REAL_ MPI_FLOAT
#else
#define _MPI_REAL_ MPI_DOUBLE
#endif


class GridMPI : public NodeBlock
{
    protected:

    int myrank, mypeindex[3], pesize[3];
    int nbrrank[6];
    int periodic[3];
    int blocksize[3];

    double gextent[3];

    MPI_Comm cart_world;

    void _get_nbr_ranks()
    {
        /* *
         * Returns ranks of the neighbor processes in the (periodic) 3D cartesian
         * topology in nbrrank = {xrank_l, xrank_r, yrank_l, yrank_r, zrank_l,
         * zrank_r}
         * */
        int coords_l[3], coords_r[3];
        for (int i = 0; i < 3; ++i)
        {
            const int im1 = (i-1 + 3) % 3;
            const int ip1 = (i+1 + 3) % 3;
            coords_l[i]   = (mypeindex[i]-1 + pesize[i]) % pesize[i];
            coords_l[im1] = mypeindex[im1];
            coords_l[ip1] = mypeindex[ip1];
            coords_r[i]   = (mypeindex[i]+1 + pesize[i]) % pesize[i];
            coords_r[im1] = mypeindex[im1];
            coords_r[ip1] = mypeindex[ip1];
            MPI_Cart_rank(cart_world, coords_l, &nbrrank[i*2 + 0]);
            MPI_Cart_rank(cart_world, coords_r, &nbrrank[i*2 + 1]);
        }
    }


    public:

    GridMPI(const int npeX, const int npeY, const int npeZ, const double maxextent = 1):
        NodeBlock()
    {
        NodeBlock::clear();

        blocksize[0] = NodeBlock::sizeX;
        blocksize[1] = NodeBlock::sizeY;
        blocksize[2] = NodeBlock::sizeZ;

        periodic[0] = 1; // only one (big) block, no smaller units
        periodic[1] = 1;
        periodic[2] = 1;

        pesize[0] = npeX; // dimension of MPI cartesian topology
        pesize[1] = npeY;
        pesize[2] = npeZ;

        h = maxextent / (std::max(pesize[0]*blocksize[0], std::max(pesize[1]*blocksize[1], pesize[2]*blocksize[2])));

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        assert(npeX*npeY*npeZ == world_size);

        MPI_Cart_create(MPI_COMM_WORLD, 3, pesize, periodic, true, &cart_world);

        MPI_Comm_rank(cart_world, &myrank);
        MPI_Cart_coords(cart_world, myrank, 3, mypeindex);

        _get_nbr_ranks();

        origin[0] += h * blocksize[0] * mypeindex[0];
        origin[1] += h * blocksize[1] * mypeindex[1];
        origin[2] += h * blocksize[2] * mypeindex[2];

        extent[0] = h * blocksize[0];
        extent[1] = h * blocksize[1];
        extent[2] = h * blocksize[2];

        gextent[0] = extent[0] * pesize[0];
        gextent[1] = extent[1] * pesize[1];
        gextent[2] = extent[2] * pesize[2];
    }

    ~GridMPI() { }


    inline int getBlocksPerDimension(int idim) const
    {
        assert(idim>=0 && idim<3);
        return pesize[idim];
    }

    inline void peindex(int mypeindex[3]) const
    {
        for(int i=0; i<3; ++i)
            mypeindex[i] = this->mypeindex[i];
    }

    inline void getNeighborRanks(int nbr[6]) const
    {
        for(int i=0; i<6; ++i)
            nbr[i] = this->nbrrank[i];
    }

    inline MPI_Comm getCartComm() const
    {
        return cart_world;
    }

    inline double getH() const
    {
        return h_gridpoint();
    }

    inline void get_gextent(double gE[3]) const
    {
        gE[0] = gextent[0];
        gE[1] = gextent[1];
        gE[2] = gextent[2];
    }
};

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
#include "NodeBlock.h"

using namespace std;

#define _MPI_REAL_ MPI_FLOAT
#define _mymax(a,b) ((a) > (b) ? (a) : (b))

/* #include "StencilInfo.h" */
/* #include "SynchronizerMPI.h" */

class GridMPI : public NodeBlock
{
    size_t timestamp;

    protected:

    /* friend class SynchronizerMPI; */

    int myrank, mypeindex[3], pesize[3];
    int nbrrank[6];
    int periodic[3];
    int blocksize[3];

    const double gextent;

    /* map<StencilInfo, SynchronizerMPI *> SynchronizerMPIs; */

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

    void _send_receive(Real *ghost_l, Real *ghost_r, const int Ng, const int left, const int right)
    {
        vector<Real> recv_buffer_l(Ng);
        vector<Real> recv_buffer_r(Ng);

        MPI_Status status;
        MPI_Sendrecv(ghost_l, Ng, _MPI_REAL_, left,  1, &recv_buffer_r[0], Ng, _MPI_REAL_, right, MPI_ANY_TAG, cart_world, &status);
        MPI_Sendrecv(ghost_r, Ng, _MPI_REAL_, right, 2, &recv_buffer_l[0], Ng, _MPI_REAL_, left,  MPI_ANY_TAG, cart_world, &status);

        //copy back
        for (int i = 0; i < Ng; ++i)
        {
            ghost_l[i] = recv_buffer_l[i];
            ghost_r[i] = recv_buffer_r[i];
        }
    }


    public:

    GridMPI(const int npeX, const int npeY, const int npeZ, const double maxextent = 1):
        NodeBlock(), gextent(maxextent), timestamp(0)
    {
        NodeBlock::clear();

        blocksize[0] = NodeBlock::sizeX;
        blocksize[1] = NodeBlock::sizeY;
        blocksize[2] = NodeBlock::sizeZ;

        periodic[0] = 1;
        periodic[1] = 1;
        periodic[2] = 1;

        pesize[0] = npeX;
        pesize[1] = npeY;
        pesize[2] = npeZ;

        const double h = maxextent / (_mymax(pesize[0]*blocksize[0], _mymax(pesize[1]*blocksize[1], pesize[2]*blocksize[2])) - 1);
        _set_gridspacing(h);

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        assert(npeX*npeY*npeZ == world_size);

        MPI_Cart_create(MPI_COMM_WORLD, 3, pesize, periodic, true, &cart_world);

        MPI_Comm_rank(cart_world, &myrank);
        MPI_Cart_coords(cart_world, myrank, 3, mypeindex);

        _get_nbr_ranks();

        const double offset = bextent + h;
        for (int i = 0; i < 3; ++i)
            origin[i] += offset * mypeindex[i];
    }

    ~GridMPI()
    {
        /* for( map<StencilInfo, SynchronizerMPI*>::const_iterator it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it) */
        /*     delete it->second; */

        /* SynchronizerMPIs.clear(); */
    }


    void send_receive_all()
    {
        const int Ng = size_ghost();

        for (int i = 0; i < nPrim; ++i)
        {
            _send_receive(xghost_l[i], xghost_r[i], Ng, nbrrank[0], nbrrank[1]);
            _send_receive(yghost_l[i], yghost_r[i], Ng, nbrrank[2], nbrrank[3]);
            _send_receive(zghost_l[i], zghost_r[i], Ng, nbrrank[4], nbrrank[5]);
        }
    }
/* // send and receive */
/* template <typename T> */
/* void send_receive(T *ghost_l, T *ghost_r, const int Ng, const int myrank, const int left, const int right) */
/* { */
/*     vector<T> recv_buffer_l(Ng); */
/*     vector<T> recv_buffer_r(Ng); */

/*     MPI_Status status; */
/*     MPI_Sendrecv(ghost_l, Ng, _MPI_DATA_TYPE_, left,  1, &recv_buffer_r[0], Ng, _MPI_DATA_TYPE_, right, MPI_ANY_TAG, cart_world, &status); */
/*     MPI_Sendrecv(ghost_r, Ng, _MPI_DATA_TYPE_, right, 2, &recv_buffer_l[0], Ng, _MPI_DATA_TYPE_, left,  MPI_ANY_TAG, cart_world, &status); */

/*     //copy back */
/*     for (int i = 0; i < Ng; ++i) */
/*     { */
/*         ghost_l[i] = recv_buffer_l[i]; */
/*         ghost_r[i] = recv_buffer_r[i]; */
/*     } */
/* } */

/* template <typename T> */
/* void send_receive_all(NodeBlock& myblock, const int myrank, const int mycoords[3], const int dims[3]) */
/* { */
/*     int nbr[6]; */
/*     get_nbr_ranks(nbr, mycoords, dims); */

/*     const int Ng = myblock.size_ghost(); */

/*     send_receive<T>(myblock.pxghost_l()[0], myblock.pxghost_r()[0], Ng, myrank, nbr[0], nbr[1]); */
/*     send_receive<T>(myblock.pyghost_l()[0], myblock.pyghost_r()[0], Ng, myrank, nbr[2], nbr[3]); */
/*     send_receive<T>(myblock.pzghost_l()[0], myblock.pzghost_r()[0], Ng, myrank, nbr[4], nbr[5]); */
/* } */

    /* vector<BlockInfo> getBlocksInfo() const */
    /* { */
    /*     return cached_blockinfo; */
    /* } */

    /* vector<BlockInfo> getResidentBlocksInfo() const */
    /* { */
    /*     return TGrid::getBlocksInfo(); */
    /* } */

    /* virtual bool avail(int ix, int iy=0, int iz=0) const */
    /* { */
    /*     //return true; */
    /*     const int originX = mypeindex[0]*mybpd[0]; */
    /*     const int originY = mypeindex[1]*mybpd[1]; */
    /*     const int originZ = mypeindex[2]*mybpd[2]; */

    /*     const int nX = pesize[0]*mybpd[0]; */
    /*     const int nY = pesize[1]*mybpd[1]; */
    /*     const int nZ = pesize[2]*mybpd[2]; */

    /*     ix = (ix + nX) % nX; */
    /*     iy = (iy + nY) % nY; */
    /*     iz = (iz + nZ) % nZ; */

    /*     const bool xinside = (ix>= originX && ix<nX); */
    /*     const bool yinside = (iy>= originY && iy<nY); */
    /*     const bool zinside = (iz>= originZ && iz<nZ); */

    /*     assert(TGrid::avail(ix-originX, iy-originY, iz-originZ)); */
    /*     return xinside && yinside && zinside; */
    /* } */

    /* inline Block& operator()(int ix, int iy=0, int iz=0) const */
    /* { */
    /*     //assuming ix,iy,iz to be global */
    /*     const int originX = mypeindex[0]*mybpd[0]; */
    /*     const int originY = mypeindex[1]*mybpd[1]; */
    /*     const int originZ = mypeindex[2]*mybpd[2]; */

    /*     const int nX = pesize[0]*mybpd[0]; */
    /*     const int nY = pesize[1]*mybpd[1]; */
    /*     const int nZ = pesize[2]*mybpd[2]; */

    /*     ix = (ix + nX) % nX; */
    /*     iy = (iy + nY) % nY; */
    /*     iz = (iz + nZ) % nZ; */

    /*     assert(ix>= originX && ix<nX); */
    /*     assert(iy>= originY && iy<nY); */
    /*     assert(iz>= originZ && iz<nZ); */

    /*     return TGrid::operator()(ix-originX, iy-originY, iz-originZ); */
    /* } */

    /* template<typename Processing> */
    /* SynchronizerMPI& sync(Processing& p) */
    /* { */
    /*     const StencilInfo stencil = p.stencil; */
    /*     assert(stencil.isvalid()); */

    /*     SynchronizerMPI * queryresult = NULL; */

    /*     typename map<StencilInfo, SynchronizerMPI*>::iterator itSynchronizerMPI = SynchronizerMPIs.find(stencil); */

    /*     if (itSynchronizerMPI == SynchronizerMPIs.end()) */
    /*     { */
    /*         queryresult = new SynchronizerMPI(SynchronizerMPIs.size(), stencil, getBlocksInfo(), cartcomm, mybpd, blocksize); */

    /*         SynchronizerMPIs[stencil] = queryresult; */
    /*     } */
    /*     else  queryresult = itSynchronizerMPI->second; */

    /*     queryresult->sync(sizeof(typename Block::element_type)/sizeof(Real), sizeof(Real)>4 ? MPI_DOUBLE : MPI_FLOAT, timestamp); */

    /*     timestamp++; */

    /*     return *queryresult; */
    /* } */

    /* template<typename Processing> */
    /* const SynchronizerMPI& get_SynchronizerMPI(Processing& p) const */
    /* { */
    /*     assert((SynchronizerMPIs.find(p.stencil) != SynchronizerMPIs.end())); */

    /*     return *SynchronizerMPIs.find(p.stencil)->second; */
    /* } */

    /* int getResidentBlocksPerDimension(int idim) const */
    /* { */
    /*     assert(idim>=0 && idim<3); */
    /*     return 1; */
    /* } */

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

    inline size_t getTimeStamp() const
    {
        return timestamp;
    }

    inline MPI_Comm getCartComm() const
    {
        return cart_world;
    }

    inline double getH() const
    {
        return h_gridpoint();
    }
};

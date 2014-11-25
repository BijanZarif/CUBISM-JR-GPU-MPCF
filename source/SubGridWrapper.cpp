/* File        : SubGridWrapper.cpp */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Wed 24 Sep 2014 02:12:49 PM CEST */
/* Modified    : Tue 07 Oct 2014 12:10:21 PM CEST */
/* Description : */
#include "SubGridWrapper.h"
#include <cstdio>
#include <cstdlib>

GridMPI* SubGridWrapper::SubBlock::supergrid = 0;

int SubGridWrapper::SubBlock::sizeX = 0;
int SubGridWrapper::SubBlock::sizeY = 0;
int SubGridWrapper::SubBlock::sizeZ = 0;
double SubGridWrapper::SubBlock::extent_x = 0.0;
double SubGridWrapper::SubBlock::extent_y = 0.0;
double SubGridWrapper::SubBlock::extent_z = 0.0;
double SubGridWrapper::SubBlock::h = 0.0;

void SubGridWrapper::make_submesh(GridMPI *grid, const int ncX, const int ncY, const int ncZ)
{
    if (_BLOCKSIZEX_ % ncX != 0)
    {
        fprintf(stderr, "ERROR: subcellsX must be an integer multiple of _BLOCKSIZEX_");
        abort();
    }
    if (_BLOCKSIZEY_ % ncY != 0)
    {
        fprintf(stderr, "ERROR: subcellsY must be an integer multiple of _BLOCKSIZEY_");
        abort();
    }
    if (_BLOCKSIZEZ_ % ncZ != 0)
    {
        fprintf(stderr, "ERROR: subcellsZ must be an integer multiple of _BLOCKSIZEZ_");
        abort();
    }

    G = grid;
    nblocks[0] = _BLOCKSIZEX_ / ncX;
    nblocks[1] = _BLOCKSIZEY_ / ncY;
    nblocks[2] = _BLOCKSIZEZ_ / ncZ;

    const double h_ = grid->getH();
    SubGridWrapper::SubBlock::sizeX = ncX;
    SubGridWrapper::SubBlock::sizeY = ncY;
    SubGridWrapper::SubBlock::sizeZ = ncZ;
    SubGridWrapper::SubBlock::extent_x = ncX * h_;
    SubGridWrapper::SubBlock::extent_y = ncY * h_;
    SubGridWrapper::SubBlock::extent_z = ncZ * h_;
    SubGridWrapper::SubBlock::h = h_;
    SubGridWrapper::SubBlock::assign_supergrid(grid);

    const int N = nblocks[0] * nblocks[1] * nblocks[2];
    blocks.clear();
    blocks.reserve(N);
    blocks.resize(N); // destroys NUMA touch! just sayin'

    double O[3];
    grid->get_origin(O);

    int mypeidx[3];
    grid->peindex(mypeidx);

#pragma omp parallel for
    for (int i=0; i < N; ++i)
    {
        const int bix = i % nblocks[0];
        const int biy = (i/nblocks[0]) % nblocks[1];
        const int biz = i/(nblocks[0]*nblocks[1]);

        const double thisOrigin[3] = {
            O[0] + bix * SubGridWrapper::SubBlock::extent_x,
            O[1] + biy * SubGridWrapper::SubBlock::extent_y,
            O[2] + biz * SubGridWrapper::SubBlock::extent_z };

        const int thisIndex[3] = {
            nblocks[0]*mypeidx[0] + bix,
            nblocks[1]*mypeidx[1] + biy,
            nblocks[2]*mypeidx[2] + biz};

        SubBlock thisBlock(thisOrigin, thisIndex);
        blocks[i] = thisBlock;
    }
}

SubGridWrapper::~SubGridWrapper()
{
    G = 0;
    SubGridWrapper::SubBlock::assign_supergrid(0);
    blocks.clear();
}

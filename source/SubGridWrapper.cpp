/* File        : SubGridWrapper.cpp */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Wed 24 Sep 2014 02:12:49 PM CEST */
/* Modified    : Thu 25 Sep 2014 03:38:15 PM CEST */
/* Description : */
#include "SubGridWrapper.h"
#include <cstdio>
#include <cstdlib>

GridMPI* SubGridWrapper::SubBlock::supergrid = 0;

uint_t SubGridWrapper::SubBlock::sizeX = 0;
uint_t SubGridWrapper::SubBlock::sizeY = 0;
uint_t SubGridWrapper::SubBlock::sizeZ = 0;
double SubGridWrapper::SubBlock::extent_x = 0.0;
double SubGridWrapper::SubBlock::extent_y = 0.0;
double SubGridWrapper::SubBlock::extent_z = 0.0;
double SubGridWrapper::SubBlock::h = 0.0;

void SubGridWrapper::make_submesh(GridMPI *grid, const uint_t ncX, const uint_t ncY, const uint_t ncZ)
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

    blocks.reserve(nblocks[0] * nblocks[1] * nblocks[2]);

    double O[3];
    grid->get_origin(O);
    for (int biz=0; biz < nblocks[2]; ++biz)
        for (int biy=0; biy < nblocks[1]; ++biy)
            for (int bix=0; bix < nblocks[0]; ++bix)
            {
                const double thisOrigin[3] = {
                    O[0] + bix * SubGridWrapper::SubBlock::extent_x,
                    O[1] + biy * SubGridWrapper::SubBlock::extent_y,
                    O[2] + biz * SubGridWrapper::SubBlock::extent_z };

                const uint_t thisIndex[3] = {bix, biy, biz};

                SubBlock thisBlock(thisOrigin, thisIndex);
                blocks.push_back(thisBlock);
            }

    SubGridWrapper::SubBlock::assign_supergrid(grid);
}

SubGridWrapper::~SubGridWrapper()
{
    SubGridWrapper::SubBlock::assign_supergrid(0);
    blocks.clear();
}

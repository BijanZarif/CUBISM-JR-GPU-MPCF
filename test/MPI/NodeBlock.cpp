/* *
 * NodeBlock.cpp
 *
 * Created by Fabian Wermelinger on 6/19/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "NodeBlock.h"
#include <stdlib.h>
#include <cmath>
using namespace std;


#ifndef _ALIGNBYTES_
#define _ALIGNBYTES_ 16
#endif

static void _allocate_aligned(void **memptr, size_t alignment, size_t bytes)
{
    const int retval = posix_memalign(memptr, alignment, bytes);
    assert(retval == 0);
}

void NodeBlock::_alloc()
{
    const int N = sizeX * sizeY * sizeZ;
    const int Ng = 3 * sizeX * sizeX; //only for cubic block!
    for (int prim = 0; prim < nPrim; ++prim)
    {
        _allocate_aligned((void **)&data[prim], max(8, _ALIGNBYTES_), sizeof(Real) * N);
        _allocate_aligned((void **)&tmp[prim],  max(8, _ALIGNBYTES_), sizeof(Real) * N);

        _allocate_aligned((void **)&xghost_l[prim],  max(8, _ALIGNBYTES_), sizeof(Real) * Ng);
        _allocate_aligned((void **)&xghost_r[prim],  max(8, _ALIGNBYTES_), sizeof(Real) * Ng);
        _allocate_aligned((void **)&yghost_l[prim],  max(8, _ALIGNBYTES_), sizeof(Real) * Ng);
        _allocate_aligned((void **)&yghost_r[prim],  max(8, _ALIGNBYTES_), sizeof(Real) * Ng);
        _allocate_aligned((void **)&zghost_l[prim],  max(8, _ALIGNBYTES_), sizeof(Real) * Ng);
        _allocate_aligned((void **)&zghost_r[prim],  max(8, _ALIGNBYTES_), sizeof(Real) * Ng);
    }
}

void NodeBlock::_dealloc()
{
    for (int prim = 0; prim < nPrim; ++prim)
    {
        free(data[prim]);
        free(tmp[prim]);

        free(xghost_l[prim]);
        free(xghost_r[prim]);
        free(yghost_l[prim]);
        free(yghost_r[prim]);
        free(zghost_l[prim]);
        free(zghost_r[prim]);
    }
}

void NodeBlock::clear_data()
{
    const int N = sizeX * sizeY * sizeZ;
    for (int prim = 0; prim < nPrim; ++prim)
    {
        Real *pdata = data[prim];
        for (int i = 0; i < N; ++i)
            pdata[i] = static_cast<Real>(0.0);
    }
}

void NodeBlock::clear_tmp()
{
    const int N = sizeX * sizeY * sizeZ;
    for (int prim = 0; prim < nPrim; ++prim)
    {
        Real *ptmp = tmp[prim];
        for (int i = 0; i < N; ++i)
            ptmp[i] = static_cast<Real>(0.0);
    }
}

void NodeBlock::get_pos(const unsigned int ix, const unsigned int iy, const unsigned int iz, double pos[3]) const
{
    pos[0] = origin[0] + h * ix;
    pos[1] = origin[1] + h * iy;
    pos[2] = origin[2] + h * iz;
}

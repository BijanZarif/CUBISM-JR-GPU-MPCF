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
    for (int var = 0; var < NVAR; ++var)
    {
        _allocate_aligned((void **)&data[var], max(8, _ALIGNBYTES_), sizeof(Real) * N);
        _allocate_aligned((void **)&tmp[var],  max(8, _ALIGNBYTES_), sizeof(Real) * N);
    }
}

void NodeBlock::_dealloc()
{
    for (int var = 0; var < NVAR; ++var)
    {
        free(data[var]);
        free(tmp[var]);
    }
}

void NodeBlock::clear_data()
{
    const int N = sizeX * sizeY * sizeZ;
    for (int var = 0; var < NVAR; ++var)
    {
        Real *pdata = data[var];
        for (int i = 0; i < N; ++i)
            pdata[i] = static_cast<Real>(0.0);
    }
}

void NodeBlock::clear_tmp()
{
    const int N = sizeX * sizeY * sizeZ;
    for (int var = 0; var < NVAR; ++var)
    {
        Real *ptmp = tmp[var];
        for (int i = 0; i < N; ++i)
            ptmp[i] = static_cast<Real>(0.0);
    }
}

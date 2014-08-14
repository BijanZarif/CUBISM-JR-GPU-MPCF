/* File        : GPU.cuh */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Thu 14 Aug 2014 09:06:16 AM CEST */
/* Modified    : Thu 14 Aug 2014 02:49:04 PM CEST */
/* Description : GPU only, shared */
#pragma once

#include "GPU.h" // includes Types.h
#include <vector>

#define _NUM_GPU_BUF_ 2
#define _NUM_STREAMS_ 3

#define NX _BLOCKSIZEX_
#define NY _BLOCKSIZEY_
#define NXP1 NX+1
#define NYP1 NY+1

enum { VSIZE = NodeBlock::NVAR };

struct GPU_COMM
{
    // ghosts
    real_vector_t d_xgl;
    real_vector_t d_xgr;
    real_vector_t d_ygl;
    real_vector_t d_ygr;

    // GPU output
    real_vector_t d_divF;

    // GPU input (3D Arrays)
    std::vector<cudaArray_t> d_GPUin;

    GPU_COMM() : d_xgl(VSIZE,NULL), d_xgr(VSIZE,NULL), d_ygl(VSIZE,NULL), d_ygr(VSIZE,NULL), d_divF(VSIZE,NULL), d_GPUin(VSIZE,NULL) { }
};


struct DevicePointer // 7 fluid quantities
{
    // helper structure to pass compound flow variables as one kernel argument
    Real * __restrict__ r;
    Real * __restrict__ u;
    Real * __restrict__ v;
    Real * __restrict__ w;
    Real * __restrict__ e;
    Real * __restrict__ G;
    Real * __restrict__ P;
    DevicePointer(real_vector_t& c) : r(c[0]), u(c[1]), v(c[2]), w(c[3]), e(c[4]), G(c[5]), P(c[6]) { assert(c.size() == VSIZE); }
};

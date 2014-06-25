/*
 *  Convection_CUDA.h
 *  MPCFcore
 *
 *  Created by Fabian Wermelinger on 5/22/14.
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "Types.h"


class Convection_CUDA
{
public:

    const Real a, dtinvh; //LSRK3-related "a" factor, and "lambda"

    //the only constructor for this class
    Convection_CUDA(const Real a, const Real dtinvh);

    //main method of the class, it evaluates the convection term of the RHS
    void compute(const unsigned int BSX_GPU, const unsigned int BSY_GPU, const unsigned int CHUNK_WIDTH, const unsigned int global_iz);
};


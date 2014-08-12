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

    /* const Real a, dtinvh; //LSRK3-related "a" factor, and "lambda" */

    //the only constructor for this class
    Convection_CUDA() { }

    //main method of the class, it evaluates the convection term of the RHS
    void compute(const uint_t nslices, const uint_t global_iz, const int s_id);
};


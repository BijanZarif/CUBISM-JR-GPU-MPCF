/* File        : Update_QPX.h */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Fri 12 Sep 2014 11:34:13 PM CEST */
/* Modified    : Sat 27 Sep 2014 09:20:11 AM CEST */
/* Description : Copyright © 2014 ETH Zurich. All Rights Reserved. */
#pragma once

#include "Update_CPP.h"

class Update_QPX : public Update_CPP
{
public:
    Update_QPX(const Real a, const Real b, const Real dtinvh) : Update_CPP(a,b,dtinvh) { }

    void compute(real_vector_t& src, real_vector_t& tmp, real_vector_t& divF, const uint_t offset, const uint_t N);
    void state(real_vector_t& src, const uint_t offset, const uint_t N);
};

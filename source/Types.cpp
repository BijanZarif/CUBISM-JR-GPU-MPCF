/* *
 * Types.cpp
 *
 * Created by Fabian Wermelinger on 7/19/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "Types.h"

FluidElement operator*(const Real s, const FluidElement& e)
{
    FluidElement prod(e);
    prod.rho *= s; prod.u *= s; prod.v *= s; prod.w *= s; prod.energy *= s; prod.G *= s; prod.P *= s;
    return prod;
}

Real MaterialDictionary::gamma1 = 1.4;
Real MaterialDictionary::gamma2 = 1.4;
Real MaterialDictionary::pc1 = 0.0;
Real MaterialDictionary::pc2 = 0.0;

double SimTools::EPSILON;

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

// global material properties, sometimes handy to access like this. Defaults to
// air, deviating values are set in the _set_constants() method in the sim
// class
Real MaterialDictionary::rho1 = 1.0;
Real MaterialDictionary::rho2 = 1.0;
Real MaterialDictionary::gamma1 = 1.4;
Real MaterialDictionary::gamma2 = 1.4;
Real MaterialDictionary::pc1 = 0.0;
Real MaterialDictionary::pc2 = 0.0;

// this is set in the steady state sim class, every other sim must inherit from
// the steady state
double SimTools::EPSILON;

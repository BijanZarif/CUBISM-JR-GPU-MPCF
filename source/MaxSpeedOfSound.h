/*
 *  MaxSpeedOfSound.h
 *  MPCFcore
 *
 *  Created by Diego Rossinelli on 6/15/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <cstdio>

#include "Types.h"

class MaxSpeedOfSound_CPP
{
public:
    Real compute(const RealPtrVec_t& src) const;
};

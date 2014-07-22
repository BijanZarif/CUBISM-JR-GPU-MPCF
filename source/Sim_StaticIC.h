/* *
 * Sim_StaticIC.h
 *
 * Created by Fabian Wermelinger on 7/22/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include "Sim_SteadyStateMPI.h"
#include <string>

class Sim_StaticIC : public Sim_SteadyStateMPI
{
    uint_t dims[3];
    std::string filename;

    protected:
        virtual void _setup();
        virtual void _ic();

    public:
        Sim_StaticIC(const int argc, const char **argv, const int isroot);

        virtual void run() { _setup(); }
};

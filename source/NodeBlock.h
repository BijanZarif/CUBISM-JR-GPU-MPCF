/* *
 * NodeBlock.h
 *
 * Created by Fabian Wermelinger on 6/19/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <vector>


#ifdef _FLOAT_PRECISION_
typedef float Real;
#else
typedef double Real;
#endif


class NodeBlock
{
    public:

        enum PRIM
        {
            R = 0, U, V, W, E, G, P,
            NPRIMITIVES = 7
        };

        // Dimensions
        static const int NVAR = NPRIMITIVES;
        static const int sizeX = _BLOCKSIZEX_;
        static const int sizeY = _BLOCKSIZEY_;
        static const int sizeZ = _BLOCKSIZEZ_;


    protected:

        // spatial measures
        double origin[3];
        double bextent;
        double h;
        inline void _set_origin(const double O[3])
        {
            for (int i = 0; i < 3; ++i)
                origin[i] = O[i];
        }

        // Fluid data and tmp storage
        std::vector<Real *> data;
        std::vector<Real *> tmp;

        inline unsigned int _linaccess(const unsigned int ix, const unsigned int iy, const unsigned int iz)
        {
            assert(ix < sizeX);
            assert(iy < sizeY);
            assert(iz < sizeZ);
            return (ix + sizeX * (iy + sizeY * iz));
        }


    private:
        void _alloc();
        void _dealloc();


    public:

        NodeBlock(const double e_ = 1.0)
            :
                //origin{0.0, 0.0, 0.0}, // nvcc does not like this
                bextent(e_), data(NVAR, NULL), tmp(NVAR, NULL)
        {
            h = bextent / (std::max(_BLOCKSIZEX_, std::max(_BLOCKSIZEY_, _BLOCKSIZEZ_)) - 1);
            origin[0] = origin[1] = origin[2] = 0.0;
            _alloc();
        }

        virtual ~NodeBlock() { _dealloc(); }

        void clear_data();
        void clear_tmp();
        inline void clear()
        {
            clear_data();
            clear_tmp();
        }

        inline double block_extent() const { return bextent; }
        inline double h_gridpoint() const { return h; }
        void get_pos(const unsigned int ix, const unsigned int iy, const unsigned int iz, double pos[3]) const;

        inline const std::vector<Real *>& pdata() const { return data; }
        inline const std::vector<Real *>& ptmp()  const { return tmp; }
        inline std::vector<Real *>& pdata() { return data; }
        inline std::vector<Real *>& ptmp()  { return tmp; }

        inline Real& operator()(const unsigned int ix, const unsigned int iy, const unsigned int iz, const PRIM p)
        {
            return data[p][_linaccess(ix, iy, iz)];
        }
};

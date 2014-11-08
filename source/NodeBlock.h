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
        double extent[3];
        double h;

        // Fluid data and tmp storage (low storage Runge-Kutta)
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

        NodeBlock(const double maxextent = 1.0)
            :
                //origin{0.0, 0.0, 0.0}, // nvcc does not like this
                data(NVAR, NULL), tmp(NVAR, NULL)
        {
            h = maxextent / (std::max(_BLOCKSIZEX_, std::max(_BLOCKSIZEY_, _BLOCKSIZEZ_)));
            origin[0] = origin[1] = origin[2] = 0.0;
            extent[0] = h * _BLOCKSIZEX_;
            extent[1] = h * _BLOCKSIZEY_;
            extent[2] = h * _BLOCKSIZEZ_;
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

        inline double h_gridpoint() const { return h; }
        inline void get_pos(const unsigned int ix, const unsigned int iy, const unsigned int iz, double pos[3]) const
        {
            // local position, relative to origin, cell center
            pos[0] = origin[0] + h * (ix+0.5);
            pos[1] = origin[1] + h * (iy+0.5);
            pos[2] = origin[2] + h * (iz+0.5);
        }
        inline void get_origin(double O[3]) const
        {
            O[0] = origin[0];
            O[1] = origin[1];
            O[2] = origin[2];
        }
        inline void get_extent(double E[3]) const
        {
            E[0] = extent[0];
            E[1] = extent[1];
            E[2] = extent[2];
        }

        inline const std::vector<Real *>& pdata() const { return data; }
        inline const std::vector<Real *>& ptmp()  const { return tmp; }
        inline std::vector<Real *>& pdata() { return data; }
        inline std::vector<Real *>& ptmp()  { return tmp; }

        inline size_t size() const { return _BLOCKSIZEX_ * _BLOCKSIZEY_ * _BLOCKSIZEZ_; }

        inline Real& operator()(const unsigned int ix, const unsigned int iy, const unsigned int iz, const PRIM p)
        {
            return data[p][_linaccess(ix, iy, iz)];
        }
};

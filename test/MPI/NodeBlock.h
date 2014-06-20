/* *
 * NodeBlock.h
 *
 * Created by Fabian Wermelinger on 6/19/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include <stdlib.h>
#include <assert.h>
#include <vector>

#ifndef _BLOCKSIZE_
#define _BLOCKSIZE_ 16
#endif

#ifdef _FLOAT_PRECISION_
typedef float Real;
#else
typedef double Real;
#endif


class NodeBlock //cubic block of data designated for a single node
{
    public:

        enum PRIM
        {
            R = 0, U, V, W, E, G, P,
            NPRIMITIVES = 7
        };

        // Dimensions
        static const int nPrim = NPRIMITIVES;
        static const int sizeX = _BLOCKSIZE_;
        static const int sizeY = _BLOCKSIZE_;
        static const int sizeZ = _BLOCKSIZE_;


    protected:

        // spatial measures
        double origin[3];
        double bextent; //since cubic block, all sides = extent
        double h;
        inline void _set_gridspacing(const double h_)
        {
            h = h_;
            bextent = h_ * (_BLOCKSIZE_-1);
        }
        inline void _set_origin(const double O[3])
        {
            for (int i = 0; i < 3; ++i)
                origin[i] = O[i];
        }


        // Fluid data and tmp storage
        std::vector<Real *> data;
        std::vector<Real *> tmp;

        inline const unsigned int _linaccess(const unsigned int ix, const unsigned int iy, const unsigned int iz)
        {
            assert(0 <= ix && ix < sizeX);
            assert(0 <= iy && iy < sizeY);
            assert(0 <= iz && iz < sizeZ);
            return (ix + sizeX * (iy + sizeY * iz));
        }


        // Ghost buffers -> MOVE TO GPUProcessing!!
        std::vector<Real *> xghost_l, xghost_r;
        std::vector<Real *> yghost_l, yghost_r;
        std::vector<Real *> zghost_l, zghost_r;



    private:
        void _alloc();
        void _dealloc();


    public:

        NodeBlock(const double e_ = 1.0) :
            origin{0.0, 0.0, 0.0},
            bextent(e_), h(e_ / (_BLOCKSIZE_ - 1)),
            data(nPrim, NULL), tmp(nPrim, NULL),
            xghost_l(nPrim, NULL), xghost_r(nPrim, NULL),
            yghost_l(nPrim, NULL), yghost_r(nPrim, NULL),
            zghost_l(nPrim, NULL), zghost_r(nPrim, NULL) { _alloc(); }
        virtual ~NodeBlock() { _dealloc(); }

        void clear_data();
        void clear_tmp();
        inline void clear()
        {
            clear_data();
            clear_tmp();
        }

        inline int size() const { return sizeX * sizeY * sizeZ; }
        inline int size_ghost() const { return 3* sizeY * sizeZ; } //only for cubic block!
        inline double block_extent() const { return bextent; }
        inline double h_gridpoint() const { return h; }
        void get_pos(const unsigned int ix, const unsigned int iy, const unsigned int iz, double pos[3]) const;

        inline const std::vector<Real *>& pdata() const { return data; }
        inline const std::vector<Real *>& ptmp()  const { return tmp; }
        inline const std::vector<Real *>& pxghost_l() const { return xghost_l; }
        inline const std::vector<Real *>& pxghost_r() const { return xghost_r; }
        inline const std::vector<Real *>& pyghost_l() const { return yghost_l; }
        inline const std::vector<Real *>& pyghost_r() const { return yghost_r; }
        inline const std::vector<Real *>& pzghost_l() const { return zghost_l; }
        inline const std::vector<Real *>& pzghost_r() const { return zghost_r; }

        inline Real& operator()(const unsigned int ix, const unsigned int iy, const unsigned int iz, const PRIM p)
        {
            Real * const ptr = data[p];
            return ptr[_linaccess(ix, iy, iz)];
        }
};

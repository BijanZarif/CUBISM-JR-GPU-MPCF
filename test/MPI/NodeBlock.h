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

typedef float Real;


class NodeBlock
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

        // Fluid data and tmp storage
        std::vector<Real *> data;
        std::vector<Real *> tmp;

        // Ghost buffers
        std::vector<Real *> xghost_l, xghost_r;
        std::vector<Real *> yghost_l, yghost_r;
        std::vector<Real *> zghost_l, zghost_r;


    private:
        void _alloc();
        void _dealloc();


    public:

        NodeBlock() :
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

        inline const std::vector<Real *>& pdata() const { return data; }
        inline const std::vector<Real *>& ptmp()  const { return tmp; }
        inline const std::vector<Real *>& pxghost_l() const { return xghost_l; }
        inline const std::vector<Real *>& pxghost_r() const { return xghost_r; }
        inline const std::vector<Real *>& pyghost_l() const { return yghost_l; }
        inline const std::vector<Real *>& pyghost_r() const { return yghost_r; }
        inline const std::vector<Real *>& pzghost_l() const { return zghost_l; }
        inline const std::vector<Real *>& pzghost_r() const { return zghost_r; }
};

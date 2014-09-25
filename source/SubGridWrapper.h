/* File        : SubGridWrapper.h */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Wed 24 Sep 2014 01:51:15 PM CEST */
/* Modified    : Thu 25 Sep 2014 02:10:49 PM CEST */
/* Description : Emulates smaller sub-blocks, based on GridMPI. */
#pragma once

#include "GridMPI.h"
#include "Types.h"

#include <vector>



class SubGridWrapper
{
public:
    class SubBlock
    {
        const double origin[3];
        const uint_t block_index[3];
        GridMPI& grid;

    public:
        static uint_t sizeX;
        static uint_t sizeY;
        static uint_t sizeZ;
        static double extent_x;
        static double extent_y;
        static double extent_z;
        static double h;

        SubBlock(const double O[3], const uint_t idx[3], GridMPI& G) : origin{O[0], O[1], O[2]}, block_index{idx[0], idx[1], idx[2]}, grid(G) { }

        inline void get_pos(const unsigned int ix, const unsigned int iy, const unsigned int iz, Real pos[3]) const
        {
            // local position, relative to origin, cell center
            pos[0] = origin[0] + h * (ix+0.5);
            pos[1] = origin[1] + h * (iy+0.5);
            pos[2] = origin[2] + h * (iz+0.5);
        }
        inline void get_index(const unsigned int lix, const unsigned int liy, const unsigned int liz, uint_t gidx[3]) const
        {
            // returns global grid cell index for a process
            gidx[0] = block_index[0] * sizeX + lix;
            gidx[1] = block_index[1] * sizeY + liy;
            gidx[2] = block_index[2] * sizeZ + liz;
        }
        inline void get_origin(double O[3]) const
        {
            O[0] = origin[0];
            O[1] = origin[1];
            O[2] = origin[2];
        }
        inline uint_t get_block_index(const uint_t i) const { assert(i<3); return block_index[i]; }

        void set(const int lix, const int liy, const int liz, const FluidElement& IC)
        {
            const int ix = block_index[0] * sizeX + lix;
            const int iy = block_index[1] * sizeY + liy;
            const int iz = block_index[2] * sizeZ + liz;

            grid(ix, iy, iz, GridMPI::PRIM::R) = IC.rho;
            grid(ix, iy, iz, GridMPI::PRIM::U) = IC.u;
            grid(ix, iy, iz, GridMPI::PRIM::V) = IC.v;
            grid(ix, iy, iz, GridMPI::PRIM::W) = IC.w;
            grid(ix, iy, iz, GridMPI::PRIM::E) = IC.energy;
            grid(ix, iy, iz, GridMPI::PRIM::G) = IC.G;
            grid(ix, iy, iz, GridMPI::PRIM::P) = IC.P;
        }

        FluidElement operator()(const int lix, const int liy, const int liz) const
        {
            const int ix = block_index[0] * sizeX + lix;
            const int iy = block_index[1] * sizeY + liy;
            const int iz = block_index[2] * sizeZ + liz;

            FluidElement ret;

            ret.rho    = grid(ix, iy, iz, GridMPI::PRIM::R);
            ret.u      = grid(ix, iy, iz, GridMPI::PRIM::U);
            ret.v      = grid(ix, iy, iz, GridMPI::PRIM::V);
            ret.w      = grid(ix, iy, iz, GridMPI::PRIM::W);
            ret.energy = grid(ix, iy, iz, GridMPI::PRIM::E);
            ret.G      = grid(ix, iy, iz, GridMPI::PRIM::G);
            ret.P      = grid(ix, iy, iz, GridMPI::PRIM::P);

            return ret;
        }
    };


private:
    /* const uint_t nblock_x, nblock_y, nblock_z; */
    uint_t nblocks[3];
    std::vector<SubBlock *> blocks;

    GridMPI* G;

public:
    SubGridWrapper() { }
    ~SubGridWrapper();

    void mesh(GridMPI *grid, const uint_t ncX, const uint_t ncY, const uint_t ncZ);

    inline SubBlock *operator[](const int block_id) { return blocks[block_id]; }
    inline size_t size() const { return blocks.size(); }

    // adapts CUBISM interface
    inline const std::vector<SubBlock*>& getBlocksInfo() const { return blocks; }
    inline double getH() const { return G->getH(); }
    inline void peindex(int mypeindex[3]) const { G->peindex(mypeindex); }
    inline MPI_Comm getCartComm() const { return G->getCartComm(); }


    int getResidentBlocksPerDimension(const int idim) const
    {
        assert(idim>=0 && idim<3);
        return nblocks[idim];
    }

    int getBlocksPerDimension(const int idim) const
    {
        assert(idim>=0 && idim<3);
        return nblocks[idim] * G->getBlocksPerDimension(idim);
    }
};

typedef SubGridWrapper::SubBlock MenialBlock;

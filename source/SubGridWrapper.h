/* File        : SubGridWrapper.h */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Wed 24 Sep 2014 01:51:15 PM CEST */
/* Modified    : Tue 07 Oct 2014 11:47:04 AM CEST */
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
        double origin[3];
        int block_index[3]; // local to MPI process
        int block_index_universe[3]; // global

        static GridMPI* supergrid;

    public:
        static int sizeX;
        static int sizeY;
        static int sizeZ;
        static double extent_x;
        static double extent_y;
        static double extent_z;
        static double h;

        static void assign_supergrid(GridMPI* const G) { supergrid = G; }

        SubBlock() {}
        SubBlock(const double O[3], const int blk_idx[3], const int universe_idx[3])
        {
            for (int i=0; i < 3; ++i)
            {
                origin[i] = O[i];
                block_index[i] = blk_idx[i];
                block_index_universe[i] = universe_idx[i];
            }
        }

        inline void get_pos(const int lix, const int liy, const int liz, Real pos[3]) const
        {
            // local position, relative to origin, cell center
            pos[0] = origin[0] + h * (lix+0.5);
            pos[1] = origin[1] + h * (liy+0.5);
            pos[2] = origin[2] + h * (liz+0.5);
        }
        inline void get_index(const int lix, const int liy, const int liz, uint_t gidx[3]) const
        {
            // returns MPI grid cell index for a process
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
        inline int get_block_index(const uint_t i) const { assert(i<3); return block_index[i]; }
        inline int get_block_index_universe(const uint_t i) const { assert(i<3); return block_index_universe[i]; }

        void set(const int lix, const int liy, const int liz, const FluidElement& IC)
        {
            // compute MPI index from  local index ix
            const int ix = block_index[0] * sizeX + lix;
            const int iy = block_index[1] * sizeY + liy;
            const int iz = block_index[2] * sizeZ + liz;

            (*supergrid)(ix, iy, iz, GridMPI::PRIM::R) = IC.rho;
            (*supergrid)(ix, iy, iz, GridMPI::PRIM::U) = IC.u;
            (*supergrid)(ix, iy, iz, GridMPI::PRIM::V) = IC.v;
            (*supergrid)(ix, iy, iz, GridMPI::PRIM::W) = IC.w;
            (*supergrid)(ix, iy, iz, GridMPI::PRIM::E) = IC.energy;
            (*supergrid)(ix, iy, iz, GridMPI::PRIM::G) = IC.G;
            (*supergrid)(ix, iy, iz, GridMPI::PRIM::P) = IC.P;
        }

        FluidElement operator()(const int lix, const int liy, const int liz) const
        {
            // compute MPI index from local index lix
            const int ix = block_index[0] * sizeX + lix;
            const int iy = block_index[1] * sizeY + liy;
            const int iz = block_index[2] * sizeZ + liz;

            FluidElement ret;

            ret.rho    = (*supergrid)(ix, iy, iz, GridMPI::PRIM::R);
            ret.u      = (*supergrid)(ix, iy, iz, GridMPI::PRIM::U);
            ret.v      = (*supergrid)(ix, iy, iz, GridMPI::PRIM::V);
            ret.w      = (*supergrid)(ix, iy, iz, GridMPI::PRIM::W);
            ret.energy = (*supergrid)(ix, iy, iz, GridMPI::PRIM::E);
            ret.G      = (*supergrid)(ix, iy, iz, GridMPI::PRIM::G);
            ret.P      = (*supergrid)(ix, iy, iz, GridMPI::PRIM::P);

            return ret;
        }
    };


private:
    int nblocks[3];
    std::vector<SubBlock> blocks;

    GridMPI* G;

public:
    SubGridWrapper() { }
    virtual ~SubGridWrapper();

    void make_submesh(GridMPI *grid, const int ncX, const int ncY, const int ncZ);

    inline SubBlock& operator[](const int block_id) { return blocks[block_id]; }
    inline const SubBlock& operator[](const int block_id) const { return blocks[block_id]; }
    inline size_t size() const { return blocks.size(); }

    // adapts CUBISM interface (a bit ugly, don't you think?)
    inline const std::vector<SubBlock>& getBlocksInfo() const { return blocks; }
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

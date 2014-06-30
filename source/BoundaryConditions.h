/*
 *  BoundaryConditions.h
 *
 *  Created by Fabian Wermelinger on 6/16/14.
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#ifdef _FLOAT_PRECISION_
typedef float Real;
#else
typedef double Real;
#endif


template<typename TGrid>
class BoundaryConditions
{
protected:

    int s[3], e[3];

    const std::vector<Real *>& pdata;
    const unsigned int startZ, deltaZ;
    typedef unsigned int (*index_map)(const int ix, const int iy, const int iz);

    template<int dir, int side>
    void _setup()
    {
        s[0] =  0;
        s[1] =  0;
        s[2] =  dir==2? 0 : startZ;

        e[0] =  dir==0? 3 : TGrid::sizeX;
        e[1] =  dir==1? 3 : TGrid::sizeY;
        e[2] =  dir==2? 3 : startZ + deltaZ;
    }


public:

    BoundaryConditions(const std::vector<Real *>& data,
            const unsigned int first_slice_iz = 0, const unsigned int n_slices = TGrid::sizeZ)
        :
            pdata(data),
            startZ(first_slice_iz), deltaZ(n_slices)
    {
        s[0]=s[1]=s[2]=0;
        e[0]=e[1]=e[2]=0;
    }

    inline Real operator()(const Real * const psrc, int ix, int iy, int iz) const
    {
        return psrc[ix + TGrid::sizeX * (iy + TGrid::sizeY * iz)];
    }

    template<int dir, int side, index_map map>
    void applyBC_absorbing(std::vector<Real *>& halo)
    {
        _setup<dir,side>();

#pragma omp parallel for
        for (int p = 0; p < TGrid::NVAR; ++p)
        {
            Real * const phalo = halo[p];
            const Real * const psrc = pdata[p];
            for(int iz=s[2]; iz<e[2]; iz++)
                for(int iy=s[1]; iy<e[1]; iy++)
                    for(int ix=s[0]; ix<e[0]; ix++)
                    {
                        // iz-startZ to operate on an arbitrary chunk
                        phalo[map(ix, iy, iz-startZ)] = (*this)(psrc,
                                dir==0? (side==0? 0:TGrid::sizeX-1):ix,
                                dir==1? (side==0? 0:TGrid::sizeY-1):iy,
                                dir==2? (side==0? 0:TGrid::sizeZ-1):iz);
                    }
        }
    }

    /* template<int dir, int side> */
    /* void applyBC_absorbing_better_faces() */
    /* { */
    /*     _setup<dir,side>(); */

    /*     //Zeroth order extrapolation for faces. */
    /*     //This fills up ALL the ghost values although corners and edges will not be right. */
    /*     //Corners and edges will be overwritten on request by void applyBC_absorbing_better_tensorials<int,int>(). */
    /*     for(int iz=s[2]; iz<e[2]; iz++) */
    /*         for(int iy=s[1]; iy<e[1]; iy++) */
    /*             for(int ix=s[0]; ix<e[0]; ix++) */
    /*             { */
    /*                 (*this)(ix,iy,iz) = (*this)(dir==0? (side==0? 0:TBlock::sizeX-1):ix, */
    /*                                             dir==1? (side==0? 0:TBlock::sizeY-1):iy, */
    /*                                             dir==2? (side==0? 0:TBlock::sizeZ-1):iz); */
    /*             } */
    /* } */

    /* void applyBC_absorbing_better_tensorials_edges() */
    /* { */
    /*     const int bsize[3] = {TBlock::sizeX, TBlock::sizeY, TBlock::sizeZ}; */
    /*     int s[3], e[3]; */

    /*     //Edges */
    /*     { */
    /*         for(int d=0; d<3; ++d) */
    /*             for(int b=0; b<2; ++b) */
    /*                 for(int a=0; a<2; ++a) */
    /*                 { */
    /*                     const int d1 = (d + 1) % 3; */
    /*                     const int d2 = (d + 2) % 3; */

    /*                     s[d]  = stencilStart[d]; */
    /*                     s[d1] = a*(bsize[d1]-stencilStart[d1])+stencilStart[d1]; */
    /*                     s[d2] = b*(bsize[d2]-stencilStart[d2])+stencilStart[d2]; */

    /*                     e[d]  = bsize[d]-1+stencilEnd[d]; */
    /*                     e[d1] = a*(bsize[d1]-1+stencilEnd[d1]); */
    /*                     e[d2] = b*(bsize[d2]-1+stencilEnd[d2]); */

    /*                     for(int iz=s[2]; iz<e[2]; iz++) */
    /*                         for(int iy=s[1]; iy<e[1]; iy++) */
    /*                             for(int ix=s[0]; ix<e[0]; ix++) */
    /*                             { */
    /*                                 (*this)(ix,iy,iz) = d==0? (*this)(ix,a*(bsize[1]-1),b*(bsize[2]-1)) : (d==1? (*this)(a*(bsize[0]-1),iy,b*(bsize[2]-1)) : (*this)(a*(bsize[0]-1),b*(bsize[1]-1),iz)); */
    /*                             } */
    /*                 } */
    /*     } */
    /* } */

    /* void applyBC_absorbing_better_tensorials_corners() */
    /* { */
    /*     const int bsize[3] = {TBlock::sizeX, TBlock::sizeY, TBlock::sizeZ}; */
    /*     int s[3], e[3]; */

    /*     //Corners */
    /*     { */
    /*         for(int c=0; c<2; ++c) */
    /*             for(int b=0; b<2; ++b) */
    /*                 for(int a=0; a<2; ++a) */
    /*                 { */
    /*                     s[0]  = a*(bsize[0]-stencilStart[0])+stencilStart[0]; */
    /*                     s[1] =  b*(bsize[1]-stencilStart[1])+stencilStart[1]; */
    /*                     s[2] =  c*(bsize[2]-stencilStart[2])+stencilStart[2]; */

    /*                     e[0]  = a*(bsize[0]-1+stencilEnd[0]); */
    /*                     e[1] =  b*(bsize[1]-1+stencilEnd[1]); */
    /*                     e[2] =  c*(bsize[2]-1+stencilEnd[2]); */

    /*                     for(int iz=s[2]; iz<e[2]; iz++) */
    /*                         for(int iy=s[1]; iy<e[1]; iy++) */
    /*                             for(int ix=s[0]; ix<e[0]; ix++) */
    /*                             { */
    /*                                 (*this)(ix,iy,iz) = (*this)(a*(bsize[0]-1),b*(bsize[1]-1),c*(bsize[2]-1)); */
    /*                             } */
    /*                 } */
    /*     } */
    /* } */

    /* template<int dir, int side> */
    /* void applyBC_reflecting() */
    /* { */
    /*     _setup<dir,side>(); */

    /*     for(int iz=s[2]; iz<e[2]; iz++) */
    /*         for(int iy=s[1]; iy<e[1]; iy++) */
    /*             for(int ix=s[0]; ix<e[0]; ix++) */
    /*             { */
    /*                 TElement source = (*this)(dir==0? (side==0? -ix-1:2*TBlock::sizeX-ix-1):ix, */
    /*                                           dir==1? (side==0? -iy-1:2*TBlock::sizeY-iy-1):iy, */
    /*                                           dir==2? (side==0? -iz-1:2*TBlock::sizeZ-iz-1):iz); */

    /*                 (*this)(ix,iy,iz).rho      = source.rho; */
    /*                 (*this)(ix,iy,iz).u        = ((dir==0)? -1:1)*source.u; */
    /*                 (*this)(ix,iy,iz).v        = ((dir==1)? -1:1)*source.v; */
    /*                 (*this)(ix,iy,iz).w        = ((dir==2)? -1:1)*source.w; */
    /*                 (*this)(ix,iy,iz).energy   = source.energy; */
    /*                 (*this)(ix,iy,iz).G = source.G; */
    /*                 (*this)(ix,iy,iz).P = source.P; */
    /*             } */
    /* } */

    /* template<int dir, int side> */
    /* void applyBC_dirichlet(const TElement& p) */
    /* { */
    /*     _setup<dir,side>(); */

    /*     for(int iz=s[2]; iz<e[2]; iz++) */
    /*         for(int iy=s[1]; iy<e[1]; iy++) */
    /*             for(int ix=s[0]; ix<e[0]; ix++) */
    /*             { */
    /*                 (*this)(ix,iy,iz).rho      = p.rho; */
    /*                 (*this)(ix,iy,iz).u        = p.u; */
    /*                 (*this)(ix,iy,iz).v        = p.v; */
    /*                 (*this)(ix,iy,iz).w        = p.w; */
    /*                 (*this)(ix,iy,iz).energy   = p.energy; */
    /*                 (*this)(ix,iy,iz).G        = p.G; */
    /*                 (*this)(ix,iy,iz).P        = p.P; */
    /*             } */
    /* } */

    /* template<int dir, int side> */
    /* void applyBC_spaceDirichlet(const TElement& p, const Real t, const Real h) */
    /* { */
    /*     _setup<dir,side>(); */

    /*     for(int iz=s[2]; iz<e[2]; iz++) */
    /*         for(int iy=s[1]; iy<e[1]; iy++) */
    /*             for(int ix=s[0]; ix<e[0]; ix++) */
    /*             { */
    /*                 Real pc = 0; */
    /*                 (*this)(ix,iy,iz).rho      = p.rho; */
    /*                 (*this)(ix,iy,iz).u        = p.u; */
    /*                 (*this)(ix,iy,iz).v        = p.v; */
    /*                 (*this)(ix,iy,iz).w        = p.w; */
    /*                 (*this)(ix,iy,iz).G        = p.G; */
    /*                 pc = p.P; */
    /*                 (*this)(ix,iy,iz).P        = p.P; */
    /*                 /// time is scaled by tc = R0/c_L, R0=0.1 equivalent to 50microns, domain size: 500microns */
    /*                 Real c_L                 = sqrt((1/p.G+1)*(pc+10)/p.rho);//speed of sound */
    /*                 Real p_ratio             = 1000; */
    /*                 Real radius              = 50e-6; */
    /*                 Real time_scale          = 0.1/c_L*1500/radius; //time scale between us and real dimension */
    /*                 Real t_off               = (ix<0? abs(ix): ix-TBlock::sizeX+1)*h/c_L;//space offset */
    /*                 //printf("t: %e, t_off: %e, c_L: %f\n",t,t_off,c_L*10); */
    /*                 //(*this)(ix,iy,iz).energy   = _pulse((t+t_off)/time_scale,p_ratio)*p.G+pc; */
    /*                 (*this)(ix,iy,iz).energy = p.energy; */
    /*             } */
    /* } */

    /* template<int dir, int side> */
    /* void applyBC_absorbing_subsonic(const TElement& p, const Real h) */
    /* { */
    /*     _setup<dir,side>(); */

    /*     for(int iz=s[2]; iz<e[2]; iz++) */
    /*         for(int iy=s[1]; iy<e[1]; iy++) */
    /*             for(int ix=s[0]; ix<e[0]; ix++) */
    /*             { */
    /*                 (*this)(ix,iy,iz) = (*this)(dir==0? (side==0? 0:TBlock::sizeX-1):ix, */
    /*                                             dir==1? (side==0? 0:TBlock::sizeY-1):iy, */
    /*                                             dir==2? (side==0? 0:TBlock::sizeZ-1):iz); */

    /*                 const Real ke = 0.5*(pow((*this)(ix,iy,iz).u,2)+pow((*this)(ix,iy,iz).v,2)+pow((*this)(ix,iy,iz).w,2))/(*this)(ix,iy,iz).rho; */

    /*                 const int offset = dir==0? (side==0? abs(ix) : ix-s[0]+1) : dir==1? (side==0? abs(iy) : iy-s[1]+1) : (side==0? abs(iz) : iz-s[2]+1); */

    /*                 (*this)(ix,iy,iz).energy = 0.25*(p.energy+ke-(*this)(dir==0? (side==0? 0:TBlock::sizeX-1):ix, */
    /*                                                                      dir==1? (side==0? 0:TBlock::sizeY-1):iy, */
    /*                                                                      dir==2? (side==0? 0:TBlock::sizeZ-1):iz).energy)*(double)offset + (*this)(dir==0? (side==0? 0:TBlock::sizeX-1):ix, */
    /*                                                                                                                                                dir==1? (side==0? 0:TBlock::sizeY-1):iy, */
    /*                                                                                                                                                dir==2? (side==0? 0:TBlock::sizeZ-1):iz).energy; */
    /*             } */
    /* } */
};

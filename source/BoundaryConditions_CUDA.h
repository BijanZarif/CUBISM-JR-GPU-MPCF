/*
 *  BoundaryConditions_CUDA.h
 *  MPCFnode
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

typedef float Real;


template<typename TBlock, typename TElement>
class BoundaryCondition_CUDA
{
protected:

    int s[3], e[3];
    int stencilStart[3], stencilEnd[3];

    const float * const src;
    const unsigned int gptfloats;
    const unsigned int startZ, deltaZ;
    typedef unsigned int (*index_function)(const int ix, const int iy, const int iz);

    template<int dir, int side>
    void _setup()
    {
        s[0] =  dir==0? (side==0? stencilStart[0]: TBlock::sizeX) : 0;
        s[1] =  dir==1? (side==0? stencilStart[1]: TBlock::sizeY) : 0;
        s[2] =  dir==2? (side==0? stencilStart[2]: TBlock::sizeZ) : startZ;

        e[0] =  dir==0? (side==0? 0: TBlock::sizeX + stencilEnd[0]-1) : TBlock::sizeX;
        e[1] =  dir==1? (side==0? 0: TBlock::sizeY + stencilEnd[1]-1) : TBlock::sizeY;
        e[2] =  dir==2? (side==0? 0: TBlock::sizeZ + stencilEnd[2]-1) : startZ + deltaZ;
    }

    /* Real _pulse(const Real t_star, const Real p_ratio) */
    /* { */
    /*     const Real Pa = p_ratio*2.38*10;//target peak pressure (factor 2.38 if no velocity) */
    /*     const Real omega = 2*M_PI*0.5/6e-6;//tensile part set to be 6microseconds (6e-6) */
    /*     const Real rise  = 1.03*(1-exp(-9.21e7*t_star));//50ns rise time */
    /*     const Real alpha = 9.1e5; */

    /*     Real p = rise*2*Pa*exp(-alpha*t_star)*cos(omega*t_star + M_PI/3.); */

    /*     return p; */
    /* } */

public:

    BoundaryCondition_CUDA(const int ss[3], const int se[3], const float * const src0,
            const unsigned int first_slice_iz = 0, const unsigned int n_slices = TBlock::sizeZ)
        :
            src(src0),
            gptfloats(TBlock::gptfloats),
            startZ(first_slice_iz), deltaZ(n_slices)
    {
        s[0]=s[1]=s[2]=0;
        e[0]=e[1]=e[2]=0;

        stencilStart[0] = ss[0];
        stencilStart[1] = ss[1];
        stencilStart[2] = ss[2];

        stencilEnd[0] = se[0];
        stencilEnd[1] = se[1];
        stencilEnd[2] = se[2];
    }

    TElement& operator()(int ix, int iy, int iz)
    {
        // assumes one single block!
        const unsigned int idx = gptfloats * (ix + TBlock::sizeX * (iy + TBlock::sizeY * iz));
        return * (TElement*) &src[idx];
    }

    template<int dir, int side>
    void applyBC_absorbing(std::vector<float*> dst, index_function idx)
    {
        _setup<dir,side>();

        for(int iz=s[2]; iz<e[2]; iz++)
            for(int iy=s[1]; iy<e[1]; iy++)
                for(int ix=s[0]; ix<e[0]; ix++)
                {
                    const TElement& b = (*this)(dir==0? (side==0? 0:TBlock::sizeX-1):ix,
                                                dir==1? (side==0? 0:TBlock::sizeY-1):iy,
                                                dir==2? (side==0? 0:TBlock::sizeZ-1):iz);

                    // startZ is subtracted from iz because dst can be a ghost
                    // buffer for a particular subdomain (local index
                    // coordinates)
                    const unsigned int ghost_id = idx(ix, iy, iz-startZ);
                    (dst[0])[ghost_id] = b.rho;
                    (dst[1])[ghost_id] = b.u;
                    (dst[2])[ghost_id] = b.v;
                    (dst[3])[ghost_id] = b.w;
                    (dst[4])[ghost_id] = b.energy;
                    (dst[5])[ghost_id] = b.G;
                    (dst[6])[ghost_id] = b.P;
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
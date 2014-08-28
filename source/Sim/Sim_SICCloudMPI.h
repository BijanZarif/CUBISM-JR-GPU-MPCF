/* *
 * Sim_SICCloudMPI.h
 *
 * Created by Fabian Wermelinger on 7/22/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include "Sim_SteadyStateMPI.h"
#include <fstream>
#include <sstream>
#include <mpi.h>
#include <iostream>

namespace SICCloudData
{
    extern double seed_s[3], seed_e[3];
    extern double min_rad, max_rad;
    extern int n_shapes, n_small, small_count;
    extern int n_sensors;
    extern Real nx, ny, nz, Sx, Sy, Sz;
    extern Real pressureRatio, rho0, c0, u0, p0, rhoB, uB, pB;
    extern Real rho1, u1, p1, mach;
    extern Real g1, g2, pc1, pc2;
}

///////////////////////////////////////////////////////////////////////////////
// This stuff is taken from CUBISM-MPCF Test_Cloud.h and adapted to this code
///////////////////////////////////////////////////////////////////////////////
//base class is a sphere
class shape
{
    protected:
        Real center[3], radius;
        Real bbox_s[3], bbox_e[3];

    public:
        shape()
        {
            const Real x_c = (Real)drand48()*(SICCloudData::seed_e[0]-SICCloudData::seed_s[0])+SICCloudData::seed_s[0];
            const Real y_c = (Real)drand48()*(SICCloudData::seed_e[1]-SICCloudData::seed_s[1])+SICCloudData::seed_s[1];

            const Real thickness = SICCloudData::seed_e[2]-SICCloudData::seed_s[2];
            const Real z_c = (Real)drand48() * thickness + SICCloudData::seed_s[2];
            const Real _center[3] = {x_c, y_c, z_c};

            const Real _radius = (Real)drand48()*(SICCloudData::max_rad-SICCloudData::min_rad)+SICCloudData::min_rad;

            radius = _radius;

            for(int i=0; i<3; ++i)
            {
                center[i] = _center[i];
                bbox_s[i] = _center[i]-_radius-1.5*SimTools::EPSILON;
                bbox_e[i] = _center[i]+_radius+1.5*SimTools::EPSILON;
            }
        }

        void set(const Real _center[3], const Real _radius)
        {
            radius = _radius;

            for(int i=0; i<3; ++i)
            {
                center[i] = _center[i];
                bbox_s[i] = _center[i]-_radius-1.5*SimTools::EPSILON;
                bbox_e[i] = _center[i]+_radius+1.5*SimTools::EPSILON;
            }
        }

        void get_bbox(double s[3], double e[3]) const
        {
            for(int i=0; i<3; ++i)
            {
                s[i] = bbox_s[i];
                e[i] = bbox_e[i];
            }
        }

        bool inside_my_box(const Real pos[3]) const
        {
            const bool bXin = pos[0]>bbox_s[0] && pos[0]<bbox_e[0];
            const bool bYin = pos[1]>bbox_s[1] && pos[1]<bbox_e[1];
            const bool bZin = pos[2]>bbox_s[2] && pos[2]<bbox_e[2];

            return bXin && bYin && bZin;
        }

        void get_center(Real c[3]) const
        {
            for(int i=0; i<3; ++i)
                c[i] = center[i];
        }

        Real get_rad() const
        {
            return radius;
        }

        Real eval(const Real pos[3]) const
        {
            return sqrt(pow(pos[0]-center[0],2)+pow(pos[1]-center[1],2)+pow(pos[2]-center[2],2))-radius;
        }

        //Every other derived shape should implement this method.
        bool rejection_check(shape * this_shape, const Real start[3], const Real end[3]) const
        {
            double s[3], e[3];
            this->get_bbox(s,e);

            //this rule checks that the buble is inside the bounding box
            const bool bOut = s[0]<start[0] || s[1]<start[1] || s[2]<start[2] ||
                e[0]>end[0] || e[1]>end[1] || e[2]>end[2];

            if (bOut)
                return true;

            if(this!=this_shape)
            {
                double this_s[3], this_e[3];

                this_shape->get_bbox(this_s,this_e);

                const Real overlap_start[3] =
                {
                    max(this_s[0], s[0]),
                    max(this_s[1], s[1]),
                    max(this_s[2], s[2])
                };

                const Real overlap_end[3] =
                {
                    min(this_e[0], e[0]),
                    min(this_e[1], e[1]),
                    min(this_e[2], e[2])
                };

                const bool bOverlap = overlap_end[0] > overlap_start[0] && overlap_end[1] > overlap_start[1] && overlap_end[2] > overlap_start[2];

                if (bOverlap)
                    return true;
            }

            return false;
        }

        Real * get_pointer_to_beginninig() {return &center[0];}

        static vector<shape> make_many(const Real h, string filename)
        {
            vector<shape> v_shapes;
            string line_content;

            ifstream f_read_cloud(filename);

            if (!f_read_cloud.good())
            {
                std::cout << "Watchout! cant read the file " << filename << ". Aborting now...\n";
                abort();
            }

            while(std::getline(f_read_cloud, line_content))
            {
                int idx;
                Real c[3], rad;
                istringstream line(line_content);
                line >> idx >> c[0] >> c[1] >> c[2] >> rad;

                /* std::cout << "shape " << idx << " " <<  c[0] << " " << c[1] << " " << c[2] << " " << rad << std::endl; */

                shape cur_shape;
                cur_shape.set(c,rad);
                v_shapes.push_back(cur_shape);

                //if (f_read_cloud.eof()) break;
            }

            f_read_cloud.close();

            return v_shapes;
        }
};

template<class Tshape=shape>
class Seed
{
    double start[3], end[3];

    vector<Tshape> v_shapes;

    public:

    Seed() { }

    Seed(const double _start[3], const double _end[3])
    {
        start[0] = _start[0];
        start[1] = _start[1];
        start[2] = _start[2];

        end[0] = _end[0];
        end[1] = _end[1];
        end[2] = _end[2];
    }

    void make_shapes(int ntocheck, string filename, const Real h)
    {
        v_shapes = Tshape::make_many(h, filename);

        std::cout << "number of shapes are " << v_shapes.size() << std::endl;

        if (v_shapes.size() != ntocheck)
        {
            std::cout << "PROBLEM! ntocheck is " << ntocheck << " which does not correspond to the number of shapes!!!\n";
        }
    }

    Seed retain_shapes(const double mystart[3], const double extent[3]) const
    {
        assert(v_shapes.size() > 0);

        const double myend[3] = {
            mystart[0] + extent[0],
            mystart[1] + extent[1],
            mystart[2] + extent[2]
        };

        Seed<Tshape> retval(mystart, myend);

        for(int i=0; i<v_shapes.size(); ++i)
        {
            Tshape curr_shape = v_shapes[i];

            double s[3],e[3];
            curr_shape.get_bbox(s,e);

            const Real xrange =  min(myend[0], e[0]) - max(mystart[0], s[0]);
            const Real yrange =  min(myend[1], e[1]) - max(mystart[1], s[1]);
            const Real zrange =  min(myend[2], e[2]) - max(mystart[2], s[2]);
            const bool bOverlap = (xrange > 0) && (yrange > 0) && (zrange > 0);

            if (bOverlap) retval.v_shapes.push_back(curr_shape);
        }

        return retval;
    }

    vector<Tshape> get_shapes() const { return v_shapes; }

    vector<Tshape>& get_shapes() { return this->v_shapes; }

    int get_shapes_size() const { return v_shapes.size(); }

    void set_shapes(vector<Tshape> v) { v_shapes = v; }
};

template<class Tshape=shape>
inline double eval(const vector<Tshape>& v_shapes, const Real pos[3])
{
    Real d = HUGE_VAL;

    for( int i=0; i<v_shapes.size(); ++i)
    {
        const Real newdistance = v_shapes[i].eval(pos);

        d = min(d, newdistance);
    }

    const double eps = SimTools::EPSILON;
    const double alpha = M_PI*min(1., max(0., (double)(d + eps)/(2 * eps)));

    return 0.5 + 0.5 * cos(alpha);
}

template<class Tshape=shape>
inline double eval_shifted(const vector<Tshape>& v_shapes, const Real pos[3])
{
    Real d = HUGE_VAL;

    for( int i=0; i<v_shapes.size(); ++i)
    {
        const Real newdistance = v_shapes[i].eval(pos);

        d = min(d, newdistance);
    }

    const double eps = SimTools::EPSILON;
    const double alpha = M_PI*min(1., max(0., (double)(d + eps)/(2 * eps)));//this 0 shift for pressure is very important.

    return 0.5 + 0.5 * cos(alpha);
}
///////////////////////////////////////////////////////////////////////////////


class Sim_SICCloudMPI : public Sim_SteadyStateMPI
{
    void _initialize_cloud();
    void _set_cloud(Seed<shape> **myseed);
    void _ic_quad(const Seed<shape> * const myseed);

    protected:
        virtual void _allocGPU();
        virtual void _ic();
        virtual void _dump(const std::string basename = "data");

    public:
        Sim_SICCloudMPI(const int argc, const char ** argv, const int isroot);
};


class GPUlabSICCloud : public GPUlab
{
    protected:
        void _apply_bc(const double t = 0)
        {
            BoundaryConditions<GridMPI> bc(grid.pdata());
            if (myFeature[0] == SKIN) bc.template applyBC_absorbing<0,0,ghostmap::X>(halox.left);
            if (myFeature[1] == SKIN) bc.template applyBC_absorbing<0,1,ghostmap::X>(halox.right);
            if (myFeature[2] == SKIN) bc.template applyBC_absorbing<1,0,ghostmap::Y>(haloy.left);
            if (myFeature[3] == SKIN) bc.template applyBC_absorbing<1,1,ghostmap::Y>(haloy.right);
            if (myFeature[4] == SKIN) bc.template applyBC_absorbing<2,0,ghostmap::Z>(haloz.left);
            if (myFeature[5] == SKIN) bc.template applyBC_absorbing<2,1,ghostmap::Z>(haloz.right);
        }

    public:
        GPUlabSICCloud(GridMPI& grid, const uint_t nslices, const int verb) : GPUlab(grid, nslices, verb) { }
};

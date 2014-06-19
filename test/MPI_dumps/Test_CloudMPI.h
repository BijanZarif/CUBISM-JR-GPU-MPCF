/*
 *  Test_CloudMPI.h
 *  MPCFcluster
 *
 *  Created by Babak Hejazialhosseini on 2/25/13.
 *  Copyright 2012 ETH Zurich. All rights reserved.
 *
 *  *
 *	PATER NOSTER, qui es in caelis, sanctificetur nomen tuum.
 *	Adveniat regnum tuum.
 *	Fiat voluntas tua, sicut in caelo et in terra.
 *	Panem nostrum quotidianum da nobis hodie,
 *	et dimitte nobis debita nostra sicut et nos dimittimus debitoribus nostris.
 *	Et ne nos inducas in tentationem,
 *	sed libera nos a malo.
 *	Amen.
 *
 */

#pragma once

#include <limits>
#include <Test_Cloud.h>
#include "Test_SICMPI.h"

#define Tshape shape

typedef BlockLabMPI< BlockLab< FluidBlock, std::allocator> > LabLaplace;

template<typename TLab>
struct GaussSeidel
{
    StencilInfo stencil;
    
    int stencil_start[3];
    int stencil_end[3];
    
    GaussSeidel(): stencil(-2, -2, -2, +3, +3, +3, false, 1, 4)
    {
        stencil_start[0] = stencil_start[1] = stencil_start[2] = -2;
        stencil_end[0] = stencil_end[1] = stencil_end[2] = 3;
    }
    
    GaussSeidel(const GaussSeidel& c): stencil(-2, -2, -2, +3, +3, +3, false, 1, 4)
    {
        stencil_start[0] = stencil_start[1] = stencil_start[2] = -2;
        stencil_end[0] = stencil_end[1] = stencil_end[2] = 3;
    }
    
    inline void operator()(TLab& lab, const BlockInfo& info, FluidBlock& o) const
    {
		
		for(int iz=0; iz<FluidBlock::sizeZ; iz++)
            for(int iy=0; iy<FluidBlock::sizeY; iy++)
                for(int ix=0; ix<FluidBlock::sizeX; ix++)
                {
                    Real p[15];
                    const int span = 5;
                    
                    for(int dir=0;dir<3;dir++)
                        for(int i=0;i<span;i++)
                            p[i+dir*span] = lab(ix+ (dir==0? i-2 : 0 ), iy+ (dir==1? i-2 : 0), iz+ (dir==2? i-2 : 0)).energy;
                    
                    Real pressure_new = 0;
                    for(int dir=0;dir<3;dir++)
                        pressure_new += p[dir*span+1]+p[dir*span+3];
                    
                    const double G1 = 1.0/(Simulation_Environment::GAMMA1-1);
                    const double G2 = 1.0/(Simulation_Environment::GAMMA2-1);
                    
                    o(ix,iy,iz).dummy = (lab(ix,iy,iz).G < 0.5*(G1+G2)) ? pressure_new/(Real)6.0 : lab(ix,iy,iz).energy;
                }
    }
};

class sensor : public shape
{
    Real origin[3], extent[3];
    
public:
    double probe, volume;
    
    sensor() : shape() { }
    
    void set(const Real _origin[3], const Real _extent[3], const Real h)
    {
        for(int i=0; i<3; ++i)
        {
            origin[i] = _origin[i];
            extent[i] = _extent[i]==0? 2*h : _extent[i];
            bbox_s[i] = origin[i];
            bbox_e[i] = origin[i]+extent[i];
        }
        
        volume = extent[0]*extent[1]*extent[2];
        probe=0;
    }
    
    static vector<sensor> make_many(const Real h, string filename="sensors.dat")
    {
        vector<sensor> v_shapes;
        
        ifstream f_read_cloud(filename);
		
		if (!f_read_cloud.good())
		{
			cout << "Watchout! cant read the file " << filename << ". Aborting now...\n";
			abort();
		}
        
        while(true)
		{
			if (!f_read_cloud.good())
				abort();
            
            int idx;
			Real origin[3], extent[3];
			f_read_cloud >> idx >> origin[0] >> origin[1] >> origin[2] >> extent[0] >> extent[1] >> extent[2];
			
			sensor cur_shape;
            cur_shape.set(origin,extent, h);
			v_shapes.push_back(cur_shape);
			
			if (f_read_cloud.eof()) break;
        }
        
        f_read_cloud.close();
        
        return v_shapes;
    }
};

template<typename Lab, typename Operator, typename TGrid>
void _process_laplace(vector<BlockInfo>& vInfo, Operator rhs, TGrid& grid, const Real t=0, bool tensorial=false)
{
#pragma omp parallel
    {
        vector<BlockInfo>myInfo = vInfo;
        BlockInfo * ary = &myInfo.front();
        
        Operator myrhs = rhs;
        Lab mylab;
        
        const SynchronizerMPI& synch = grid.get_SynchronizerMPI(myrhs);
        
        const int N = vInfo.size();
        mylab.prepare(grid, synch);
        
#pragma omp for schedule(runtime)
        for(int i=0; i<N; i++)
        {
            mylab.load(ary[i], t);
            myrhs(mylab, ary[i], *(FluidBlock*)ary[i].ptrBlock);
        }
    }
}

template <typename TGrid>
void _process_update(const vector<BlockInfo>& vInfo, TGrid& grid)
{
#pragma omp parallel
    {
        vector<BlockInfo>myInfo = vInfo;
        const int N = vInfo.size();
        
#pragma omp for schedule(runtime)
        for(int i=0; i<N; i++)
        {
			FluidBlock& b = *(FluidBlock*)myInfo[i].ptrBlock;
			
            for(int iz=0; iz<FluidBlock::sizeZ; iz++)
				for(int iy=0; iy<FluidBlock::sizeY; iy++)
					for(int ix=0; ix<FluidBlock::sizeX; ix++)
						b(ix,iy,iz).energy = b(ix,iy,iz).dummy;
        }
    }
}

class BS4
{
public:
    static inline Real eval(Real x)
    {
        const Real t = fabs(x);
        
        if (t>2) return 0;
        
        if (t>1) return pow(2-t,3)/6;
        
        return (1 + 3*(1-t)*(1 + (1-t)*(1 - (1-t))))/6;
    }
};

class Test_CloudMPI: public Test_Cloud
{
   	Test_SteadyStateMPI * t_ssmpi;
    Test_ShockBubbleMPI * t_sbmpi;
    
    void _get_rank_coords(G & grid, double mystart[3], double myextent[3])
    {
        int peidx[3];
        grid.peindex(peidx);
        const double spacing = grid.getH()*_BLOCKSIZE_;
        
        const int BPD[3] = {BPDX, BPDY, BPDZ};
        
        for(int i=0; i<3; i++)
        {
            mystart[i] = peidx[i]*BPD[i]*spacing;
            myextent[i] = BPD[i]*spacing;
        }
    }
    
    void _set_cloud()
    {
        const MPI::Cartcomm& mycartcomm = grid->getCartComm();
        //load the Cloud namespace and broadcast it
        {
            if(isroot)
                _initialize_cloud();
            
            mycartcomm.Bcast(&CloudData::n_shapes, 1, MPI::INT, 0);
            mycartcomm.Bcast(&CloudData::n_small, 1, MPI::INT, 0);
            mycartcomm.Bcast(&CloudData::small_count, 1, MPI::INT, 0);
            mycartcomm.Bcast(&CloudData::min_rad, 1, MPI::DOUBLE, 0);
            mycartcomm.Bcast(&CloudData::max_rad, 1, MPI::DOUBLE, 0);
            mycartcomm.Bcast(CloudData::seed_s, 3, MPI::DOUBLE, 0);
            mycartcomm.Bcast(CloudData::seed_e, 3, MPI::DOUBLE, 0);
            mycartcomm.Bcast(&CloudData::n_sensors, 1, MPI::INT, 0);
        }
        
        Seed<shape> myseed(CloudData::seed_s, CloudData::seed_e);
        
        //load the shapes from file and broadcast them
        {
            vector<shape> v(CloudData::n_shapes);
            
            if(isroot)
            {
                myseed.make_shapes(CloudData::n_shapes, "cloud.dat", grid->getH());
                v = myseed.get_shapes();
            }
            
            mycartcomm.Bcast(&v.front(), v.size() * sizeof(shape), MPI::CHAR, 0);
            
            if (!isroot)
                myseed.set_shapes(v);
        }
        
        assert(myseed.get_shapes_size()>0 && myseed.get_shapes_size() == CloudData::n_shapes);
        
        double mystart[3], myextent[3];
        _get_rank_coords(*grid, mystart, myextent);
        
        myseed = myseed.retain_shapes(mystart,myextent);
        MPI::COMM_WORLD.Barrier();
        
        if (isroot)
            cout << "Setting ic now...\n";
        
#ifdef _USE_HPM_
        HPM_Start("CLOUDIC");
#endif
        _my_ic_quad(*grid, myseed);
        //_my_ic(*grid, myseed);
#ifdef _USE_HPM_
        HPM_Stop("CLOUDIC");
#endif
        
        if (isroot)
            cout << "done!"<< endl;
    }
    
    void _relax_pressure(G & grid)
    {
        GaussSeidel< LabLaplace > gs;
		
        SynchronizerMPI& synch = ((G&)grid).sync(gs);
        
        while (!synch.done())
        {
            vector<BlockInfo> avail = synch.avail(1);
            
            _process_laplace< LabLaplace >(avail, gs, (G&)grid);
        }
        
        _process_update(grid.getBlocksInfo(), grid);
    }
    
    void _relax(G & grid)
    {
        if(parser("-relax").asInt(0)>0)
	    {
            if (isroot)
				cout << "relaxing pressure a little bit..."<< endl;
            
            const int Niter = parser("-relax").asInt(0);
            
            for(int i = 0; i <Niter; ++i)
            {
                if(isroot & i%100==0)
                    printf("itration %i out of %i\n", i, Niter);
                
                _relax_pressure(grid);
            }
            
            if (isroot)
				cout << "done!"<< endl;
	    }
    }
    
    void _set_energy(G & grid)
    {
        if (isroot)
            printf("Setting energy now...");
        
        Test_Cloud::_set_energy(grid);
        
        if (isroot)
            printf("done\n");
    }
    
    void _set_sensors()
    {
        if (isroot)
            printf("Setting sensors now...");
        
        {
            const MPI::Cartcomm& mycartcomm = grid->getCartComm();
            
            vector<sensor> v(CloudData::n_sensors);
            
            if(isroot)
            {
                sensors.make_shapes(CloudData::n_sensors, "sensors.dat", grid->getH());
                v = sensors.get_shapes();
            }
            
            mycartcomm.Bcast(&v.front(), v.size() * sizeof(sensor), MPI::CHAR, 0);
            
            if (!isroot)
                sensors.set_shapes(v);
            
	    }
        
        assert(sensors.get_shapes_size()>0 && sensors.get_shapes_size() == CloudData::n_sensors);
        
        double mystart[3], myextent[3];
        _get_rank_coords(*grid, mystart, myextent);
        
        present_sensors.resize(sensors.get_shapes_size(),0);
        
        const double myend[3] =
        {
            mystart[0] + myextent[0],
            mystart[1] + myextent[1],
            mystart[2] + myextent[2]
        };
        
        for(int i=0;i<sensors.get_shapes_size();i++)
        {
            double s[3],e[3];
            sensors.get_shapes()[i].get_bbox(s,e);
            
            const Real xrange =  min(myend[0], e[0]) - max(mystart[0], s[0]);
            const Real yrange =  min(myend[1], e[1]) - max(mystart[1], s[1]);
            const Real zrange =  min(myend[2], e[2]) - max(mystart[2], s[2]);
            const bool bOverlap = (xrange >= 0) && (yrange >= 0) && (zrange >= 0);
            
            if (bOverlap)
                present_sensors[i]=1;
        }
        
        if (isroot)
            printf("done\n");
        
        if (isroot)
        {
            for(int i=0;i<sensors.get_shapes_size();i++)
                printf("    %d", present_sensors[i]);
            
            printf("\n");
        }
    }
    
    void _dump_sensors(G& grid, const int step_id, const Real t, const Real dt)
    {
        vector<BlockInfo> vInfo = grid.getBlocksInfo();
        const double h =vInfo[0].h_gridpoint;
        const double h3 = h*h*h;
        const double h_proximity = h;
        
        Real p[3];
        double s[3], e[3];
        int Npoints[present_sensors.size()];
        
        for(int idx=0; idx<present_sensors.size(); idx++)
        {
            Npoints[idx] = 0;
            sensors.get_shapes()[idx].probe = 0;
            
            if(present_sensors[idx]==1)
            {
                sensors.get_shapes()[idx].get_bbox(s,e);
                
                for(int i=0; i<(int)vInfo.size(); i++)
                {
                    BlockInfo info = vInfo[i];
                    FluidBlock& b = *(FluidBlock*)info.ptrBlock;
                    
                    const Real xrange =  min(info.origin[0]+info.h, e[0]) - max(info.origin[0], s[0]);
                    const Real yrange =  min(info.origin[1]+info.h, e[1]) - max(info.origin[1], s[1]);
                    const Real zrange =  min(info.origin[2]+info.h, e[2]) - max(info.origin[2], s[2]);
                    const bool bOverlap = (xrange >= 0) && (yrange >= 0) && (zrange >= 0);
                    
                    if(bOverlap)
                        for(int iz=0; iz<FluidBlock::sizeZ; iz++)
                            for(int iy=0; iy<FluidBlock::sizeY; iy++)
                                for(int ix=0; ix<FluidBlock::sizeX; ix++)
                                {
                                    info.pos(p,ix,iy,iz);
                                    
                                    const bool bInside = p[0]>s[0] && p[0]<e[0] && p[1]>s[1] && p[1]<e[1] && p[2]>s[2] && p[2]<e[2];
                                    
                                    if ( bInside)
                                    {
                                        const double pressure = (b(ix, iy, iz).energy - 0.5/b(ix, iy, iz).rho * (b(ix, iy, iz).u*b(ix, iy, iz).u+b(ix, iy, iz).v*b(ix, iy, iz).v+b(ix, iy, iz).w*b(ix, iy, iz).w) - b(ix, iy, iz).P)/b(ix,iy,iz).G;
                                        
                                        sensors.get_shapes()[idx].probe+=pressure;
                                        Npoints[idx]++;
                                    }
                                }
                }
            }
        }
        
        double g_probe[present_sensors.size()];
        int g_Npoints[present_sensors.size()];
        
        for(int idx=0; idx<present_sensors.size(); idx++)
        {
            g_probe[idx] = 0;
            g_Npoints[idx] = 0;
            MPI::COMM_WORLD.Reduce(&sensors.get_shapes()[idx].probe, &g_probe[idx], 1, MPI::DOUBLE, MPI::SUM, 0);
            MPI::COMM_WORLD.Reduce(&Npoints[idx], &g_Npoints[idx], 1, MPI::INT, MPI::SUM, 0);
            g_probe[idx] /= (double)g_Npoints[idx];
        }
        
        if (MPI::COMM_WORLD.Get_rank()==0)
        {
            FILE * f = fopen("sensors_integrals.dat", "a");
            fprintf(f, "%d %e %e ", step_id, t, dt);
            for(int idx=0; idx<present_sensors.size(); idx++)
                fprintf(f, "%e ", g_probe[idx]);
            
            fprintf(f,"\n");
            fclose(f);
        }
        
        MPI::COMM_WORLD.Barrier();
    }
    
protected:
	int XPESIZE, YPESIZE, ZPESIZE;
    
	G * grid;
	FlowStep_LSRK3MPI<G> * stepper;
    Seed<sensor> sensors;
    vector<int> present_sensors;
    
public:
	bool isroot;
    
	Test_CloudMPI(const bool isroot, const int argc, const char ** argv):
    Test_Cloud(argc, argv), isroot(isroot)
	{
        t_ssmpi = new Test_SteadyStateMPI(isroot, argc, argv);
        t_sbmpi = new Test_ShockBubbleMPI(isroot, argc, argv);
	}
    
	void setup()
	{
		_setup_constants();
        t_ssmpi->setup_mpi_constants(XPESIZE, YPESIZE, ZPESIZE);
		
        if (!isroot)
			VERBOSITY = 0;
		
        if (isroot)
		{
			printf("////////////////////////////////////////////////////////////\n");
			printf("///////////                                      ///////////\n");
			printf("///////////               TEST Cloud MPI         ///////////\n");
			printf("///////////                                      ///////////\n");
			printf("PATER NOSTER, qui es in caelis, sanctificetur nomen tuum.\n");
			printf("Adveniat regnum tuum.\n");
			printf("Fiat voluntas tua, sicut in caelo et in terra.\n");
			printf("Panem nostrum quotidianum da nobis hodie,\n");
			printf("et dimitte nobis debita nostra sicut et\nnos dimittimus debitoribus nostris.\n");
			printf("Et ne nos inducas in tentationem,\n");
			printf("sed libera nos a malo.\n");
			printf("Amen.\n");
			printf("////////////////////////////////////////////////////////////\n");
		}
		
		const double extent = parser("-extent").asDouble(1.0);
		grid = new G(XPESIZE, YPESIZE, ZPESIZE, BPDX, BPDY, BPDZ, extent);

		assert(grid != NULL);
        
		stepper = new FlowStep_LSRK3MPI<G>(*grid, CFL, Simulation_Environment::GAMMA1, Simulation_Environment::GAMMA2, parser, VERBOSITY, &profiler,  Simulation_Environment::PC1, Simulation_Environment::PC2);
        
		if(bRESTART)
		{
			t_ssmpi->restart(*grid);
			t = t_ssmpi->get_time();
			step_id = t_ssmpi->get_stepid();
		}
		else
        {
            _set_cloud();
            
            _relax(*grid);
            
            _set_energy(*grid);
            
            if (CloudData::n_sensors>0) _set_sensors();
        }
	}
    
	void run()
	{
	    Real dt=0;
		const bool bWithIO = parser("-io").asBool("1");
		
		if (isroot) printf("HELLO RUN\n");
		bool bLoop = (NSTEPS>0) ? (step_id<NSTEPS) : (fabs(t-TEND) > std::numeric_limits<Real>::epsilon()*1e1);
        
		while(bLoop)
		{
			if (isroot) printf("Step id %d,Time %f\n", step_id, t);
			
			if (step_id % DUMPPERIOD == 0 && bWithIO)
			{
				profiler.push_start("IO WAVELET");
				t_ssmpi->vp(*grid, step_id, bVP);
				std::stringstream streamer;
				streamer<<"data-"<<step_id;
				t_ssmpi->dump(*grid, step_id, streamer.str());
				profiler.pop_stop();
			}
            
			if (step_id % SAVEPERIOD == 0 && bWithIO && step_id>0)
			{
				profiler.push_start("SAVE");
				t_ssmpi->save(*grid, step_id, t);
				profiler.pop_stop();
			}
            
#ifndef _SEQUOIA_
			if (step_id%ANALYSISPERIOD==0)
            {
			    profiler.push_start("DUMP STATISTICS");
			    t_sbmpi->dumpStatistics(*grid, step_id, t, dt);
			    if (CloudData::n_sensors>0) _dump_sensors(*grid, step_id, t, dt);
			    profiler.pop_stop();
            }
#endif
            
            profiler.push_start("STEP");
            
            if (isroot)
				printf("CFL is %f, original CFL is %f\n", stepper->CFL, CFL);
			
			dt = (*stepper)(TEND-t);
            
			profiler.pop_stop();
			
            
            if(step_id % 10 == 0 && isroot && step_id > 0)
				profiler.printSummary();
            
			t+=dt;
			step_id++;
            
			bLoop = (NSTEPS>0) ? (step_id<NSTEPS) : (fabs(t-TEND) > std::numeric_limits<Real>::epsilon()*1e1);
            
            if (dt==0)
            {
                if(bWithIO)
                {
                    std::stringstream streamer;
                    streamer<<"data-"<<step_id;
                    t_ssmpi->dump(*grid, step_id, streamer.str());
                }
                break;
            }
		}

		if (isroot)
			printf("Finishing RUN\n");
        
		return;
	}
	
	void dispose()
	{
		t_ssmpi->dispose();
		delete t_ssmpi;
		t_ssmpi = NULL;
		
		t_sbmpi->dispose();
		delete t_sbmpi;
		t_sbmpi = NULL;
		
		if (stepper != NULL)
		{
			delete stepper;
			stepper = NULL;
		}
		
		if (grid != NULL)
		{
			delete grid;
			grid = NULL;
		}
	}
};


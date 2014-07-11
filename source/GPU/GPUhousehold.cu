/* *
 * GPUhousehold.cu
 *
 * Created by Fabian Wermelinger on 6/24/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <stdio.h>
#include <vector>

#include "GPU.h" // includes Types.h

#ifdef _CUDA_TIMER_
#include "CUDA_Timer.cuh"
#endif


enum { VSIZE = NodeBlock::NVAR };

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES
///////////////////////////////////////////////////////////////////////////////
RealPtrVec_t d_tmp(VSIZE, NULL);
RealPtrVec_t d_rhs(VSIZE, NULL);
RealPtrVec_t d_xgl(VSIZE, NULL);
RealPtrVec_t d_xgr(VSIZE, NULL);
RealPtrVec_t d_ygl(VSIZE, NULL);
RealPtrVec_t d_ygr(VSIZE, NULL);

RealPtrVec_t d_xflux(VSIZE, NULL);
RealPtrVec_t d_yflux(VSIZE, NULL);
RealPtrVec_t d_zflux(VSIZE, NULL);

// extraterms for advection equations
Real *d_Gm, *d_Gp;
Real *d_Pm, *d_Pp;
Real *d_hllc_vel;
Real *d_sumG, *d_sumP, *d_divU;

// 3D arrays (GPU input)
std::vector<cudaArray_t> d_SOAin(VSIZE, NULL);

// Max SOS
int *h_maxSOS; // host, mapped
int *d_maxSOS; // device, mapped (different address)

// use non-null stream (async)
cudaStream_t stream1;
cudaStream_t stream2;
cudaStream_t stream3;

// events
cudaEvent_t divergence_completed;
cudaEvent_t update_completed;
cudaEvent_t h2d_3Darray_completed;
cudaEvent_t h2d_tmp_completed;
cudaEvent_t d2h_rhs_completed;
cudaEvent_t d2h_tmp_completed;


///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////
static void _h2d_3DArray(cudaArray_t dst, const Real * const src, const int NX, const int NY, const int NZ)
{
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent            = make_cudaExtent(NX, NY, NZ);
    copyParams.kind              = cudaMemcpyHostToDevice;
    copyParams.srcPtr            = make_cudaPitchedPtr((void *)src, NX * sizeof(Real), NX, NY);
    copyParams.dstArray          = dst;

    cudaMemcpy3DAsync(&copyParams, stream1);
}


extern "C"
{
    ///////////////////////////////////////////////////////////////////////////
    // GPU Memory alloc / dealloc
    ///////////////////////////////////////////////////////////////////////////
    void GPU::alloc(void** sos, const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t BSZ_GPU, const uint_t CHUNK_WIDTH, const bool isroot)
    {
#ifndef _MUTE_GPU_

        // processing slice size (normal to z-direction)
        const uint_t SLICE_GPU = BSX_GPU * BSY_GPU;

        // GPU output size
        const uint_t outputSize = SLICE_GPU * CHUNK_WIDTH;

        // fluxes (TODO: use one array later, process after each flux computation)
        const uint_t bSflx = (BSX_GPU+1)*BSY_GPU*CHUNK_WIDTH;
        const uint_t bSfly = BSX_GPU*(BSY_GPU+1)*CHUNK_WIDTH;
        const uint_t bSflz = BSX_GPU*BSY_GPU*(CHUNK_WIDTH+1);

        // x-/yghosts
        const uint_t xgSize = 3*BSY_GPU*CHUNK_WIDTH;
        const uint_t ygSize = BSX_GPU*3*CHUNK_WIDTH;

        // GPU allocation
        cudaChannelFormatDesc fmt = cudaCreateChannelDesc<Real>();
        for (int var = 0; var < VSIZE; ++var)
        {
            //tmp
            cudaMalloc(&d_tmp[var], outputSize*sizeof(Real));
            cudaMemset(d_tmp[var], 0, outputSize*sizeof(Real));

            // rhs
            cudaMalloc(&d_rhs[var], outputSize*sizeof(Real));
            cudaMemset(d_rhs[var], 0, outputSize*sizeof(Real));

            // fluxes
            cudaMalloc(&d_xflux[var], bSflx*sizeof(Real));
            cudaMalloc(&d_yflux[var], bSfly*sizeof(Real));
            cudaMalloc(&d_zflux[var], bSflz*sizeof(Real));
            cudaMemset(d_xflux[var], 0, bSflx*sizeof(Real));
            cudaMemset(d_yflux[var], 0, bSfly*sizeof(Real));
            cudaMemset(d_zflux[var], 0, bSflz*sizeof(Real));

            // x-/yghosts
            cudaMalloc(&d_xgl[var], xgSize*sizeof(Real));
            cudaMalloc(&d_xgr[var], xgSize*sizeof(Real));

            cudaMalloc(&d_ygl[var], ygSize*sizeof(Real));
            cudaMalloc(&d_ygr[var], ygSize*sizeof(Real));

            // GPU input SOA (+6 slices for zghosts)
            cudaMalloc3DArray(&d_SOAin[var], &fmt, make_cudaExtent(BSX_GPU, BSY_GPU, CHUNK_WIDTH+6));
        }

        // extraterm for advection (TODO: remove this uglyness!)
        cudaMalloc(&d_Gm, bSflz * sizeof(Real));
        cudaMalloc(&d_Gp, bSflz * sizeof(Real));
        cudaMalloc(&d_Pm, bSflz * sizeof(Real));
        cudaMalloc(&d_Pp, bSflz * sizeof(Real));
        cudaMalloc(&d_hllc_vel, bSflz * sizeof(Real));
        cudaMalloc(&d_sumG, outputSize * sizeof(Real));
        cudaMalloc(&d_sumP, outputSize * sizeof(Real));
        cudaMalloc(&d_divU, outputSize * sizeof(Real));

        // zero-copy maxSOS (TODO: should this be unsigned int??)
        cudaSetDeviceFlags(cudaDeviceMapHost);
        cudaHostAlloc((void**)&h_maxSOS, sizeof(int), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_maxSOS, h_maxSOS, 0);
        *(int**)sos = h_maxSOS; // return a reference to the caller

        // create streams
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);

        // create events
        cudaEventCreate(&divergence_completed);
        cudaEventCreate(&update_completed);
        cudaEventCreate(&h2d_3Darray_completed);
        cudaEventCreate(&h2d_tmp_completed);
        cudaEventCreate(&d2h_rhs_completed);
        cudaEventCreate(&d2h_tmp_completed);

        // Stats
        if (isroot)
        {
            int dev;
            cudaDeviceProp prop;
            cudaGetDevice(&dev);
            cudaGetDeviceProperties(&prop, dev);

            printf("=====================================================================\n");
            printf("[GPU ALLOCATION FOR %s]\n", prop.name);
            printf("[%5.1f MB (input SOA)]\n", VSIZE*(SLICE_GPU*(CHUNK_WIDTH+6))*sizeof(Real) / 1024. / 1024);
            printf("[%5.1f MB (tmp)]\n", VSIZE*outputSize*sizeof(Real) / 1024. / 1024);
            printf("[%5.1f MB (rhs)]\n", VSIZE*outputSize*sizeof(Real) / 1024. / 1024);
            printf("[%5.1f MB (flux storage)]\n", VSIZE*(bSflx + bSfly + bSflz)*sizeof(Real) / 1024. / 1024);
            printf("[%5.1f MB (x/yghosts)]\n", VSIZE*(xgSize + ygSize)*2*sizeof(Real) / 1024. / 1024);
            printf("[%5.1f MB (extraterm)]\n", (5*bSflx + 3*outputSize)*sizeof(Real) / 1024. / 1024);
            GPU::tell_memUsage_GPU();
            printf("=====================================================================\n");
        }
#endif
    }


    void GPU::dealloc(const bool isroot)
    {
#ifndef _MUTE_GPU_
        for (int var = 0; var < VSIZE; ++var)
        {
            // tmp
            cudaFree(d_tmp[var]);

            // rhs
            cudaFree(d_rhs[var]);

            // fluxes
            cudaFree(d_xflux[var]);
            cudaFree(d_yflux[var]);
            cudaFree(d_zflux[var]);

            // x-/yghosts
            cudaFree(d_xgl[var]);
            cudaFree(d_xgr[var]);
            cudaFree(d_ygl[var]);
            cudaFree(d_ygr[var]);

            // input SOA
            cudaFreeArray(d_SOAin[var]);
        }

        // extraterms
        cudaFree(d_Gm);
        cudaFree(d_Gp);
        cudaFree(d_Pm);
        cudaFree(d_Pp);
        cudaFree(d_hllc_vel);
        cudaFree(d_sumG);
        cudaFree(d_sumP);
        cudaFree(d_divU);

        // Max SOS
        cudaFreeHost(h_maxSOS);

        // destroy streams
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);

        // destroy events
        cudaEventDestroy(divergence_completed);
        cudaEventDestroy(update_completed);
        cudaEventDestroy(h2d_3Darray_completed);
        cudaEventDestroy(h2d_tmp_completed);
        cudaEventDestroy(d2h_rhs_completed);
        cudaEventDestroy(d2h_tmp_completed);

        // Stats
        if (isroot)
        {
            int dev;
            cudaDeviceProp prop;
            cudaGetDevice(&dev);
            cudaGetDeviceProperties(&prop, dev);

            printf("=====================================================================\n");
            printf("[FREE GPU %s]\n", prop.name);
            GPU::tell_memUsage_GPU();
            printf("=====================================================================\n");
        }
#endif
    }


    ///////////////////////////////////////////////////////////////////////////
    // H2D / D2H
    ///////////////////////////////////////////////////////////////////////////
    void GPU::upload_xy_ghosts(const uint_t Nxghost, const RealPtrVec_t& xghost_l, const RealPtrVec_t& xghost_r,
            const uint_t Nyghost, const RealPtrVec_t& yghost_l, const RealPtrVec_t& yghost_r)
    {
#ifndef _MUTE_GPU_
        // TODO: use larger arrays for ghosts to minimize API overhead +
        // increase BW performance
        tCUDA_START(stream1)
        for (int i = 0; i < VSIZE; ++i)
        {
            // x
            cudaMemcpyAsync(d_xgl[i], xghost_l[i], Nxghost*sizeof(Real), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(d_xgr[i], xghost_r[i], Nxghost*sizeof(Real), cudaMemcpyHostToDevice, stream1);
            // y
            cudaMemcpyAsync(d_ygl[i], yghost_l[i], Nyghost*sizeof(Real), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(d_ygr[i], yghost_r[i], Nyghost*sizeof(Real), cudaMemcpyHostToDevice, stream1);
        }
        tCUDA_STOP(stream1, "[GPU UPLOAD X/YGHOSTS]: ")
#endif
    }


    void GPU::h2d_3DArray(const RealPtrVec_t& src, const uint_t NX, const uint_t NY, const uint_t NZ)
    {
#ifndef _MUTE_GPU_
        tCUDA_START(stream1)
        for (int i = 0; i < VSIZE; ++i)
            _h2d_3DArray(d_SOAin[i], src[i], NX, NY, NZ);
        tCUDA_STOP(stream1, "[GPU UPLOAD 3DArray]: ")
        cudaEventRecord(h2d_3Darray_completed, stream1);
#endif
    }


    void GPU::h2d_tmp(const RealPtrVec_t& src, const uint_t N)
    {
#ifndef _MUTE_GPU_
        cudaStreamWaitEvent(stream3, h2d_3Darray_completed, 0);

        tCUDA_START(stream3)
        for (int i = 0; i < VSIZE; ++i)
            cudaMemcpyAsync(d_tmp[i], src[i], N*sizeof(Real), cudaMemcpyHostToDevice, stream3);
        tCUDA_STOP(stream3, "[GPU UPLOAD TMP]: ")
        cudaEventRecord(h2d_tmp_completed, stream3);
#endif
    }


    void GPU::d2h_rhs(RealPtrVec_t& dst, const uint_t N)
    {
#ifndef _MUTE_GPU_
        cudaStreamWaitEvent(stream2, divergence_completed, 0);

        // copy content of d_rhs to host, using the stream2 (after divergence)
        tCUDA_START(stream2)
        for (int i = 0; i < VSIZE; ++i)
            cudaMemcpyAsync(dst[i], d_rhs[i], N*sizeof(Real), cudaMemcpyDeviceToHost, stream2);
        tCUDA_STOP(stream2, "[GPU DOWNLOAD RHS]: ")
        cudaEventRecord(d2h_rhs_completed, stream2);
#endif
    }


    void GPU::d2h_tmp(RealPtrVec_t& dst, const uint_t N)
    {
#ifndef _MUTE_GPU_
        cudaStreamWaitEvent(stream2, update_completed, 0);

        // copy content of d_tmp to host, using the stream1
        tCUDA_START(stream1)
        for (int i = 0; i < VSIZE; ++i)
            cudaMemcpyAsync(dst[i], d_tmp[i], N*sizeof(Real), cudaMemcpyDeviceToHost, stream2);
        tCUDA_STOP(stream1, "[GPU DOWNLOAD TMP]: ")
        cudaEventRecord(d2h_tmp_completed, stream2);
#endif
    }


    ///////////////////////////////////////////////////////////////////////////
    // Sync
    ///////////////////////////////////////////////////////////////////////////
    void GPU::h2d_3DArray_wait()
    {
#ifndef _MUTE_GPU_
        // wait until h2d_3DArray has finished
        cudaEventSynchronize(h2d_3Darray_completed);
#endif
    }


    void GPU::d2h_rhs_wait()
    {
#ifndef _MUTE_GPU_
        // wait until d2h_rhs has finished
        cudaEventSynchronize(d2h_rhs_completed);
#endif
    }


    void GPU::d2h_tmp_wait()
    {
#ifndef _MUTE_GPU_
        // wait until d2h_tmp has finished
        cudaEventSynchronize(d2h_tmp_completed);
#endif
    }


    void GPU::syncGPU()
    {
#ifndef _MUTE_GPU_
        cudaDeviceSynchronize();
#endif
    }


    void GPU::syncStream(streamID s)
    {
#ifndef _MUTE_GPU_
        switch (s)
        {
            case S1: cudaStreamSynchronize(stream1); break;
            case S2: cudaStreamSynchronize(stream2); break;
        }
#endif
    }


    ///////////////////////////////////////////////////////////////////////////
    // Stats
    ///////////////////////////////////////////////////////////////////////////
    void GPU::tell_memUsage_GPU()
    {
#ifndef _MUTE_GPU_
        size_t free_byte, total_byte;
        const int status = cudaMemGetInfo(&free_byte, &total_byte);
        if (cudaSuccess != status)
        {
            printf("Hoppla! Can not get memory stats from GPU...\n");
            return;
        }
        const size_t used = total_byte - free_byte;
        printf("GPU memory usage: free = %5.1f MB, total = %5.1f MB (%5.1f MB used)\n",
                (double)free_byte / 1024 / 1024,
                (double)total_byte / 1024 / 1024,
                (double)used / 1024 / 1024);
#endif
    }


    void GPU::tell_GPU()
    {
#ifndef _MUTE_GPU_
        int dev;
        cudaDeviceProp prop;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);
        printf("Using device %d (%s)\n", dev, prop.name);
#endif
    }
}

/* *
 * GPUhousehold.cu
 *
 * Created by Fabian Wermelinger on 6/24/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <stdio.h>
#include <vector>

#include "CUDA_Timer.cuh"
#include "NodeBlock.h"
#include "GPU.h"

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

// 3D arrays
std::vector<cudaArray_t> d_SOA(VSIZE, NULL);

// Max SOS
int* h_maxSOS; // host, mapped
int* d_maxSOS; // device, mapped (different address)

// use non-null stream (async)
cudaStream_t stream1;
cudaStream_t stream2;

// events
cudaEvent_t divergence_completed;
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
    void GPU::alloc(void** sos, const uint_t BSX_GPU, const uint_t BSY_GPU, const uint_t BSZ_GPU, const uint_t CHUNK_WIDTH)
    {
        // THE FOLLOWING ASSUMES CUBIC DOAMIN
        const uint_t SLICE_GPU = BSX_GPU * BSY_GPU;

        // GPU output size
        const uint_t outputSize = SLICE_GPU * CHUNK_WIDTH;

        // Fluxes (use one array later, process after each flux dimension)
        const uint_t bSflx = (BSX_GPU+1)*BSY_GPU*CHUNK_WIDTH;
        const uint_t bSfly = BSX_GPU*(BSY_GPU+1)*CHUNK_WIDTH;
        const uint_t bSflz = BSX_GPU*BSY_GPU*(CHUNK_WIDTH+1);

        // Ghosts
        /* const uint_t xgSize = 3*BSY_GPU*BSZ_GPU; */
        /* const uint_t ygSize = BSX_GPU*3*BSZ_GPU; */
        const uint_t xgSize = 3 * SLICE_GPU;
        const uint_t ygSize = 3 * SLICE_GPU;

        // Allocate
        cudaChannelFormatDesc fmt =  cudaCreateChannelDesc<Real>();
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

            // ghosts
            cudaMalloc(&d_xgl[var], xgSize*sizeof(Real));
            cudaMalloc(&d_xgr[var], xgSize*sizeof(Real));
            cudaMalloc(&d_ygl[var], ygSize*sizeof(Real));
            cudaMalloc(&d_ygr[var], ygSize*sizeof(Real));

            // GPU input SOA
            cudaMalloc3DArray(&d_SOA[var], &fmt, make_cudaExtent(BSX_GPU, BSY_GPU, CHUNK_WIDTH+6));
        }

        // extraterm for advection
        cudaMalloc(&d_Gm, bSflz * sizeof(Real));
        cudaMalloc(&d_Gp, bSflz * sizeof(Real));
        cudaMalloc(&d_Pm, bSflz * sizeof(Real));
        cudaMalloc(&d_Pp, bSflz * sizeof(Real));
        cudaMalloc(&d_hllc_vel, bSflz * sizeof(Real));
        cudaMalloc(&d_sumG, outputSize * sizeof(Real));
        cudaMalloc(&d_sumP, outputSize * sizeof(Real));
        cudaMalloc(&d_divU, outputSize * sizeof(Real));

        // zero-copy maxSOS
        cudaSetDeviceFlags(cudaDeviceMapHost);
        cudaHostAlloc((void**)&h_maxSOS, sizeof(int), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_maxSOS, h_maxSOS, 0);
        *(int**)sos = h_maxSOS; // return a reference to the caller

        // create stream
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        // create event
        cudaEventCreate(&divergence_completed);
        cudaEventCreate(&h2d_3Darray_completed);
        cudaEventCreate(&h2d_tmp_completed);
        cudaEventCreate(&d2h_rhs_completed);
        cudaEventCreate(&d2h_tmp_completed);

        // Stats
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


    void GPU::dealloc()
    {
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

            // ghosts
            cudaFree(d_xgl[var]);
            cudaFree(d_xgr[var]);
            cudaFree(d_ygl[var]);
            cudaFree(d_ygr[var]);

            // input SOA
            cudaFreeArray(d_SOA[var]);
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

        // destroy stream
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);

        // destroy events
        cudaEventDestroy(divergence_completed);
        cudaEventDestroy(h2d_3Darray_completed);
        cudaEventDestroy(h2d_tmp_completed);
        cudaEventDestroy(d2h_rhs_completed);
        cudaEventDestroy(d2h_tmp_completed);

        // Stats
        int dev;
        cudaDeviceProp prop;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);

        printf("=====================================================================\n");
        printf("[FREE GPU %s]\n", prop.name);
        GPU::tell_memUsage_GPU();
        printf("=====================================================================\n");
    }


    ///////////////////////////////////////////////////////////////////////////
    // H2D / D2H
    ///////////////////////////////////////////////////////////////////////////
    /* void GPU::upload_ghosts(const uint_t Nghost, */
    /*         const Real* const xghost_L, const Real* const xghost_R, */
    /*         const Real* const yghost_L, const Real* const yghost_R) */
    /* { */
    /*     for (int i = 0; i < VSIZE; ++i) */
    /*     { */
    /*         cudaMemcpyAsync(d_xgl[i], &xghost_L[i*Nghost], Nghost*sizeof(Real), cudaMemcpyHostToDevice, stream1); */
    /*         cudaMemcpyAsync(d_xgr[i], &xghost_R[i*Nghost], Nghost*sizeof(Real), cudaMemcpyHostToDevice, stream1); */
    /*         cudaMemcpyAsync(d_ygl[i], &yghost_L[i*Nghost], Nghost*sizeof(Real), cudaMemcpyHostToDevice, stream1); */
    /*         cudaMemcpyAsync(d_ygr[i], &yghost_R[i*Nghost], Nghost*sizeof(Real), cudaMemcpyHostToDevice, stream1); */
    /*     } */
    /* } */


    void GPU::upload_xy_ghosts(const uint_t Nxghost, const RealPtrVec_t& xghost_l, const RealPtrVec_t& xghost_r,
            const uint_t Nyghost, const RealPtrVec_t& yghost_l, const RealPtrVec_t& yghost_r)
    {
        for (int i = 0; i < VSIZE; ++i)
        {
            // x
            cudaMemcpyAsync(d_xgl[i], xghost_l[i], Nxghost*sizeof(Real), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(d_xgr[i], xghost_r[i], Nxghost*sizeof(Real), cudaMemcpyHostToDevice, stream1);
            // y
            cudaMemcpyAsync(d_ygl[i], yghost_l[i], Nyghost*sizeof(Real), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(d_ygr[i], yghost_r[i], Nyghost*sizeof(Real), cudaMemcpyHostToDevice, stream1);
        }
    }


    void GPU::h2d_3DArray(const RealPtrVec_t& src, const uint_t NX, const uint_t NY, const uint_t NZ)
    {
        GPUtimer upload;
        upload.start(stream1);
        for (int i = 0; i < VSIZE; ++i)
            _h2d_3DArray(d_SOA[i], src[i], NX, NY, NZ);
        upload.stop(stream1);
        upload.print("[GPU UPLOAD 3DArray]: ");

        cudaEventRecord(h2d_3Darray_completed, stream1);
    }


    void GPU::h2d_tmp(const RealPtrVec_t& src, const uint_t N)
    {
        cudaStreamWaitEvent(stream2, h2d_3Darray_completed, 0);

        GPUtimer upload;
        upload.start(stream2);
        for (int i = 0; i < VSIZE; ++i)
            cudaMemcpyAsync(d_tmp[i], src[i], N*sizeof(Real), cudaMemcpyHostToDevice, stream2);
        upload.stop(stream2);
        upload.print("[GPU UPLOAD TMP]: ");

        cudaEventRecord(h2d_tmp_completed, stream2);
    }


    void GPU::d2h_rhs(RealPtrVec_t& dst, const uint_t N)
    {
        cudaStreamWaitEvent(stream2, divergence_completed, 0);

        // copy content of d_rhs to host, using the stream2 (after divergence)
        GPUtimer download;
        download.start(stream2);
        for (int i = 0; i < VSIZE; ++i)
            cudaMemcpyAsync(dst[i], d_rhs[i], N*sizeof(Real), cudaMemcpyDeviceToHost, stream2);
        download.stop(stream2);
        download.print("[GPU DOWNLOAD RHS]: ");

        cudaEventRecord(d2h_rhs_completed, stream2);
    }


    void GPU::d2h_tmp(RealPtrVec_t& dst, const uint_t N)
    {
        /* // wait until the device to host copy of the rhs has finished. This will */
        /* // hide the SOA to AOS conversion of the RHS data on the host, while the */
        /* // updated solution is copied to the host. */
        /* cudaStreamWaitEvent(stream1, d2h_rhs_completed, 0); */

        // copy content of d_tmp to host, using the stream1
        GPUtimer download;
        download.start(stream1);
        for (int i = 0; i < VSIZE; ++i)
            cudaMemcpyAsync(dst[i], d_tmp[i], N*sizeof(Real), cudaMemcpyDeviceToHost, stream2);
        download.stop(stream1);
        download.print("[GPU DOWNLOAD TMP]: ");

        cudaEventRecord(d2h_tmp_completed, stream2);
    }


    ///////////////////////////////////////////////////////////////////////////
    // Sync
    ///////////////////////////////////////////////////////////////////////////
    void GPU::h2d_3DArray_wait()
    {
        // wait until h2d_3DArray has finished
        cudaEventSynchronize(h2d_3Darray_completed);
    }


    void GPU::d2h_rhs_wait()
    {
        // wait until d2h_rhs has finished
        cudaEventSynchronize(d2h_rhs_completed);
    }


    void GPU::d2h_tmp_wait()
    {
        // wait until d2h_tmp has finished
        cudaEventSynchronize(d2h_tmp_completed);
    }


    void GPU::syncGPU()
    {
        cudaDeviceSynchronize();
    }


    void GPU::syncStream(streamID s)
    {
        switch (s)
        {
            case S1: cudaStreamSynchronize(stream1); break;
            case S2: cudaStreamSynchronize(stream2); break;
        }
    }


    ///////////////////////////////////////////////////////////////////////////
    // Stats
    ///////////////////////////////////////////////////////////////////////////
    void GPU::tell_memUsage_GPU()
    {
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
    }


    void GPU::tell_GPU()
    {
        int dev;
        cudaDeviceProp prop;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);
        printf("Using device %d (%s)\n", dev, prop.name);
    }
}

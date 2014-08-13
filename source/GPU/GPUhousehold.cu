/* *
 * GPUhousehold.cu
 *
 * Created by Fabian Wermelinger on 6/24/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "GPU.h" // includes Types.h

#include <stdio.h>
#include <vector>
#include <algorithm>
using namespace std;

enum { VSIZE = NodeBlock::NVAR };

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES
///////////////////////////////////////////////////////////////////////////////
real_vector_t d_flux(VSIZE, NULL);
real_vector_t d_xgl(VSIZE, NULL);
real_vector_t d_xgr(VSIZE, NULL);
real_vector_t d_ygl(VSIZE, NULL);
real_vector_t d_ygr(VSIZE, NULL);

// extraterms for advection equations
Real *d_Gm, *d_Gp;
Real *d_Pm, *d_Pp;
Real *d_hllc_vel;
Real *d_sumG, *d_sumP, *d_divU;

// GPU output
real_vector_t d_divF(VSIZE, NULL);

// 3D arrays (GPU input)
vector<cudaArray_t> d_GPUin(VSIZE, NULL);

// Max SOS
int *h_maxSOS; // host, mapped
int *d_maxSOS; // device, mapped (different address)

// use non-null stream (async)
#define _NUM_STREAMS_ 2
cudaStream_t *stream;

// events
#define _NUM_EVENTS_ 2
cudaEvent_t *event_h2d;
cudaEvent_t *event_d2h;


///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////
Profiler GPU::profiler; // combined CPU/GPU profiler

static void _h2d_3DArray(cudaArray_t dst, const Real * const src, const int nslices, const int s_id)
{
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent            = make_cudaExtent(NodeBlock::sizeX, NodeBlock::sizeY, nslices);
    copyParams.kind              = cudaMemcpyHostToDevice;
    copyParams.srcPtr            = make_cudaPitchedPtr((void *)src, NodeBlock::sizeX * sizeof(Real), NodeBlock::sizeX, NodeBlock::sizeY);
    copyParams.dstArray          = dst;

    cudaMemcpy3DAsync(&copyParams, stream[s_id]);
}


///////////////////////////////////////////////////////////////////////////
// GPU Memory alloc / dealloc
///////////////////////////////////////////////////////////////////////////
void GPU::alloc(void** sos, const uint_t nslices, const bool isroot)
{
#ifndef _MUTE_GPU_
    /* cudaDeviceReset(); */
    /* cudaSetDeviceFlags(cudaDeviceMapHost); */

    // processing slice size (normal to z-direction)
    const uint_t SLICE_GPU = NodeBlock::sizeX * NodeBlock::sizeY;

    // GPU output size
    const uint_t outputSize = SLICE_GPU * nslices;

    // fluxes
    const uint_t xflxSize = (NodeBlock::sizeX+1)*NodeBlock::sizeY*nslices;
    const uint_t yflxSize = NodeBlock::sizeX*(NodeBlock::sizeY+1)*nslices;
    const uint_t zflxSize = NodeBlock::sizeX*NodeBlock::sizeY*(nslices+1);
    const uint_t maxflxSize = max(xflxSize, max(yflxSize, zflxSize));

    // x-/yghosts
    const uint_t xgSize = 3*NodeBlock::sizeY*nslices;
    const uint_t ygSize = NodeBlock::sizeX*3*nslices;

    // GPU allocation
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<Real>();
    for (int var = 0; var < VSIZE; ++var)
    {
        // fluxes
        cudaMalloc(&d_flux[var], maxflxSize*sizeof(Real));

        // x-/yghosts
        cudaMalloc(&d_xgl[var], xgSize*sizeof(Real));
        cudaMalloc(&d_xgr[var], xgSize*sizeof(Real));

        cudaMalloc(&d_ygl[var], ygSize*sizeof(Real));
        cudaMalloc(&d_ygr[var], ygSize*sizeof(Real));

        // GPU output
        cudaMalloc(&d_divF[var], outputSize*sizeof(Real));

        // GPU input (+6 slices for zghosts)
        cudaMalloc3DArray(&d_GPUin[var], &fmt, make_cudaExtent(NodeBlock::sizeX, NodeBlock::sizeY, nslices+6));
    }

    // extraterm for advection
    cudaMalloc(&d_Gm, maxflxSize * sizeof(Real));
    cudaMalloc(&d_Gp, maxflxSize * sizeof(Real));
    cudaMalloc(&d_Pm, maxflxSize * sizeof(Real));
    cudaMalloc(&d_Pp, maxflxSize * sizeof(Real));
    cudaMalloc(&d_hllc_vel, maxflxSize * sizeof(Real));
    cudaMalloc(&d_sumG, outputSize * sizeof(Real));
    cudaMalloc(&d_sumP, outputSize * sizeof(Real));
    cudaMalloc(&d_divU, outputSize * sizeof(Real));

    // zero-copy maxSOS (TODO: should this be unsigned int?)
    cudaHostAlloc((void**)&h_maxSOS, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_maxSOS, h_maxSOS, 0);
    *(int**)sos = h_maxSOS; // return a reference to the caller

    // create streams
    stream = (cudaStream_t *) malloc(_NUM_STREAMS_ * sizeof(cudaStream_t));
    assert(stream != NULL);
    for (int i = 0 ; i < _NUM_STREAMS_; ++i)
        cudaStreamCreate(&stream[i]);

    // create events
    event_h2d = (cudaEvent_t *) malloc(_NUM_EVENTS_ * sizeof(cudaEvent_t));
    event_d2h = (cudaEvent_t *) malloc(_NUM_EVENTS_ * sizeof(cudaEvent_t));
    assert(event_h2d != NULL);
    assert(event_d2h != NULL);
    for (int i = 0; i < _NUM_EVENTS_; ++i)
    {
        cudaEventCreate(&event_h2d[i]);
        cudaEventCreate(&event_d2h[i]);
    }

    // Stats
    if (isroot)
    {
        int dev;
        cudaDeviceProp prop;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);

        printf("=====================================================================\n");
        printf("[GPU ALLOCATION FOR %s]\n", prop.name);
        printf("[%5.1f MB (input GPU)]\n", VSIZE*(SLICE_GPU*(nslices+6))*sizeof(Real) / 1024. / 1024);
        printf("[%5.1f MB (tmp)]\n", VSIZE*outputSize*sizeof(Real) / 1024. / 1024);
        printf("[%5.1f MB (rhs)]\n", VSIZE*outputSize*sizeof(Real) / 1024. / 1024);
        printf("[%5.1f MB (flux storage)]\n", VSIZE*(xflxSize + yflxSize + zflxSize)*sizeof(Real) / 1024. / 1024);
        printf("[%5.1f MB (x/yghosts)]\n", VSIZE*(xgSize + ygSize)*2*sizeof(Real) / 1024. / 1024);
        printf("[%5.1f MB (extraterm)]\n", (5*maxflxSize + 3*outputSize)*sizeof(Real) / 1024. / 1024);
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
        // fluxes
        cudaFree(d_flux[var]);

        // x-/yghosts
        cudaFree(d_xgl[var]);
        cudaFree(d_xgr[var]);
        cudaFree(d_ygl[var]);
        cudaFree(d_ygr[var]);

        // GPU output
        cudaFree(d_divF[var]);

        // input GPU
        cudaFreeArray(d_GPUin[var]);
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
    for (int i = 0; i < _NUM_STREAMS_; ++i)
        cudaStreamDestroy(stream[i]);
    free(stream);

    // destroy events
    for (int i = 0; i < _NUM_EVENTS_; ++i)
    {
        cudaEventDestroy(event_h2d[i]);
        cudaEventDestroy(event_d2h[i]);
    }
    free(event_h2d);
    free(event_d2h);

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
// COMMUNICATION H2D / D2H
///////////////////////////////////////////////////////////////////////////
void GPU::upload_xy_ghosts(const uint_t Nxghost, const real_vector_t& xghost_l, const real_vector_t& xghost_r,
        const uint_t Nyghost, const real_vector_t& yghost_l, const real_vector_t& yghost_r, const int s_id)
{
#ifndef _MUTE_GPU_
    assert(0 <= s_id && s_id < _NUM_STREAMS_);

    // TODO: use larger arrays for ghosts to minimize API overhead +
    // increase BW performance
    GPU::profiler.push_startCUDA("SEND GHOSTS", &stream[s_id]);
    for (int i = 0; i < VSIZE; ++i)
    {
        // x
        cudaMemcpyAsync(d_xgl[i], xghost_l[i], Nxghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]);
        cudaMemcpyAsync(d_xgr[i], xghost_r[i], Nxghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]);
        // y
        cudaMemcpyAsync(d_ygl[i], yghost_l[i], Nyghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]);
        cudaMemcpyAsync(d_ygr[i], yghost_r[i], Nyghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]);
    }
    GPU::profiler.pop_stopCUDA();
#endif
}


void GPU::h2d_3DArray(const real_vector_t& src, const uint_t nslices, const int s_id)
{
#ifndef _MUTE_GPU_
    assert(0 <= s_id && s_id < _NUM_STREAMS_);
    GPU::profiler.push_startCUDA("SEND 3DARRAY", &stream[s_id]);
    for (int i = 0; i < VSIZE; ++i)
        _h2d_3DArray(d_GPUin[i], src[i], nslices, s_id);
    GPU::profiler.pop_stopCUDA();
    cudaEventRecord(event_h2d[s_id], stream[s_id]);
#endif
}


void GPU::d2h_divF(real_vector_t& dst, const uint_t N, const int s_id)
{
#ifndef _MUTE_GPU_
    // download divF for current chunk
    GPU::profiler.push_startCUDA("RECV DIV(F)", &stream[s_id]);
    for (int i = 0; i < VSIZE; ++i)
        cudaMemcpyAsync(dst[i], d_divF[i], N*sizeof(Real), cudaMemcpyDeviceToHost, stream[s_id]);
    GPU::profiler.pop_stopCUDA();
    cudaEventRecord(event_d2h[s_id], stream[s_id]);
#endif
}


///////////////////////////////////////////////////////////////////////////
// Sync
///////////////////////////////////////////////////////////////////////////
void GPU::wait_h2d(const int e_id)
{
#ifndef _MUTE_GPU_
    cudaEventSynchronize(event_h2d[e_id]);
#endif
}


void GPU::wait_d2h(const int e_id)
{
#ifndef _MUTE_GPU_
    cudaEventSynchronize(event_d2h[e_id]);
#endif
}


void GPU::syncGPU()
{
#ifndef _MUTE_GPU_
    cudaDeviceSynchronize();
#endif
}


void GPU::syncStream(const int s_id)
{
#ifndef _MUTE_GPU_
    cudaStreamSynchronize(stream[s_id]);
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
        printf("Whoot! Can not get memory stats from GPU...\n");
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

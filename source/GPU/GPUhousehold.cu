/* *
 * GPUhousehold.cu
 *
 * Created by Fabian Wermelinger on 6/24/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "GPU.cuh"

#include <stdio.h>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES
///////////////////////////////////////////////////////////////////////////////
// flux storage (COMPUTE stream)
real_vector_t d_flux(VSIZE, NULL);

// extraterms for advection equations (COMPUTE stream)
Real *d_Gm, *d_Gp;
Real *d_Pm, *d_Pp;
Real *d_hllc_vel;
Real *d_sumG, *d_sumP, *d_divU;

// Max SOS
int *h_maxSOS; // host, mapped
int *d_maxSOS; // device, mapped (different address)

struct GPU_COMM gpu_comm[_NUM_GPU_BUF_];

// use non-null stream (async)
cudaStream_t *stream;

// events
cudaEvent_t *event_h2d;
cudaEvent_t *event_d2h;
cudaEvent_t *event_compute;


///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////
Profiler GPU::profiler; // combined CPU/GPU profiler

static void _h2d_3DArray(cudaArray_t dst, const Real * const src, const int nslices, const int s_id)
{
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent            = make_cudaExtent(NX, NY, nslices);
    copyParams.kind              = cudaMemcpyHostToDevice;
    copyParams.srcPtr            = make_cudaPitchedPtr((void *)src, NX * sizeof(Real), NX, NY);
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
    // Flux storage
    size_t computational_bytes = 0;
    for (int var = 0; var < VSIZE; ++var)
    {
        cudaMalloc(&d_flux[var], maxflxSize*sizeof(Real));
        computational_bytes += maxflxSize * sizeof(Real);
    }

    // extraterm for advection
    cudaMalloc(&d_Gm, maxflxSize * sizeof(Real));
    cudaMalloc(&d_Gp, maxflxSize * sizeof(Real));
    cudaMalloc(&d_Pm, maxflxSize * sizeof(Real));
    cudaMalloc(&d_Pp, maxflxSize * sizeof(Real));
    cudaMalloc(&d_hllc_vel, maxflxSize * sizeof(Real));
    computational_bytes += 5 * maxflxSize * sizeof(Real);
    cudaMalloc(&d_sumG, outputSize * sizeof(Real));
    cudaMalloc(&d_sumP, outputSize * sizeof(Real));
    cudaMalloc(&d_divU, outputSize * sizeof(Real));
    computational_bytes += 3 * outputSize * sizeof(Real);

    // Communication buffers
    size_t ghost_bytes  = 0;
    size_t input_bytes  = 0;
    size_t output_bytes = 0;
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<Real>();
    for (int i = 0; i < _NUM_GPU_BUF_; ++i)
    {
        GPU_COMM * const mybuf = &gpu_comm[i];
        for (int var = 0; var < VSIZE; ++var)
        {
            // x-/yghosts
            cudaMalloc(&(mybuf->d_xgl[var]), xgSize*sizeof(Real));
            cudaMalloc(&(mybuf->d_xgr[var]), xgSize*sizeof(Real));
            cudaMalloc(&(mybuf->d_ygl[var]), ygSize*sizeof(Real));
            cudaMalloc(&(mybuf->d_ygr[var]), ygSize*sizeof(Real));
            ghost_bytes += 2 * xgSize * sizeof(Real) + 2 * ygSize * sizeof(Real);

            // GPU output
            cudaMalloc(&(mybuf->d_divF[var]), outputSize*sizeof(Real));
            output_bytes += outputSize * sizeof(Real);

            // GPU input (+6 slices for zghosts)
            cudaMalloc3DArray(&(mybuf->d_GPUin[var]), &fmt, make_cudaExtent(NX, NY, nslices+6));
            input_bytes += NX * NY * (nslices+6) * sizeof(Real);
        }
    }

    // zero-copy maxSOS (TODO: should this be unsigned int?)
    cudaHostAlloc((void**)&h_maxSOS, sizeof(int), cudaHostAllocMapped);
    computational_bytes += sizeof(int);
    cudaHostGetDevicePointer(&d_maxSOS, h_maxSOS, 0);
    *(int**)sos = h_maxSOS; // return a reference to the caller

    // create streams
    stream = (cudaStream_t *) malloc(_NUM_STREAMS_ * sizeof(cudaStream_t));
    assert(stream != NULL);
    for (int i = 0 ; i < _NUM_STREAMS_; ++i)
        cudaStreamCreate(&stream[i]);

    // create events
    event_h2d     = (cudaEvent_t *) malloc(_NUM_STREAMS_ * sizeof(cudaEvent_t));
    event_d2h     = (cudaEvent_t *) malloc(_NUM_STREAMS_ * sizeof(cudaEvent_t));
    event_compute = (cudaEvent_t *) malloc(_NUM_STREAMS_ * sizeof(cudaEvent_t));
    assert(event_h2d != NULL);
    assert(event_d2h != NULL);
    assert(event_compute != NULL);
    for (int i = 0; i < _NUM_STREAMS_; ++i)
    {
        cudaEventCreate(&event_h2d[i]);
        cudaEventCreate(&event_d2h[i]);
        cudaEventCreate(&event_compute[i]);
    }

    // Stats
    if (isroot)
    {
        int dev;
        cudaDeviceProp prop;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);

        printf("=====================================================================\n");
        printf("[GPU ALLOCATION FOR %s]\n",   prop.name);
        printf("[%5.1f MB (GPU input)]\n",    input_bytes / 1024. / 1024);
        printf("[%5.1f MB (GPU ghosts)]\n",   ghost_bytes / 1024. / 1024);
        printf("[%5.1f MB (GPU output)]\n",   output_bytes / 1024. / 1024);
        printf("[%5.1f MB (Compute storage)]\n", computational_bytes / 1024. / 1024);
        GPU::tell_memUsage_GPU();
        printf("=====================================================================\n");
    }
#endif
}


void GPU::dealloc(const bool isroot)
{
#ifndef _MUTE_GPU_
    for (int var = 0; var < VSIZE; ++var)
        cudaFree(d_flux[var]);

    // extraterms
    cudaFree(d_Gm);
    cudaFree(d_Gp);
    cudaFree(d_Pm);
    cudaFree(d_Pp);
    cudaFree(d_hllc_vel);
    cudaFree(d_sumG);
    cudaFree(d_sumP);
    cudaFree(d_divU);

    for (int i = 0; i < _NUM_GPU_BUF_; ++i)
    {
        GPU_COMM * const mybuf = &gpu_comm[i];
        for (int var = 0; var < VSIZE; ++var)
        {
            // x-/yghosts
            cudaFree(mybuf->d_xgl[var]);
            cudaFree(mybuf->d_xgr[var]);
            cudaFree(mybuf->d_ygl[var]);
            cudaFree(mybuf->d_ygr[var]);

            // GPU output
            cudaFree(mybuf->d_divF[var]);

            // input GPU
            cudaFreeArray(mybuf->d_GPUin[var]);
        }
    }

    // Max SOS
    cudaFreeHost(h_maxSOS);

    // destroy streams
    for (int i = 0; i < _NUM_STREAMS_; ++i)
        cudaStreamDestroy(stream[i]);
    free(stream);

    // destroy events
    for (int i = 0; i < _NUM_STREAMS_; ++i)
    {
        cudaEventDestroy(event_h2d[i]);
        cudaEventDestroy(event_d2h[i]);
        cudaEventDestroy(event_compute[i]);
    }
    free(event_h2d);
    free(event_d2h);
    free(event_compute);

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
void GPU::h2d_input(
        const uint_t Nxghost, const real_vector_t& xghost_l, const real_vector_t& xghost_r,
        const uint_t Nyghost, const real_vector_t& yghost_l, const real_vector_t& yghost_r,
        const real_vector_t& src, const uint_t nslices,
        const uint_t gbuf_id, const int chunk_id)
{
#ifndef _MUTE_GPU_
    assert(gbuf_id < _NUM_GPU_BUF_);

    const uint_t s_id = chunk_id % _NUM_STREAMS_;
    GPU_COMM * const mybuf = &gpu_comm[gbuf_id];

    // previous stream has priority, don't interrupt
    const uint_t s_idm1 = ((chunk_id-1) + _NUM_STREAMS_) % _NUM_STREAMS_;
    assert(s_idm1 < _NUM_STREAMS_);
    cudaStreamWaitEvent(stream[s_id], event_h2d[s_idm1], 0);

    char prof_item[256];

    // TODO: use larger arrays for ghosts to minimize API overhead +
    // increase BW performance. (LOW PRIORITY)
    sprintf(prof_item, "SEND GHOSTS (%d)", s_id);
    GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
    for (int i = 0; i < VSIZE; ++i)
    {
        // x
        cudaMemcpyAsync(mybuf->d_xgl[i], xghost_l[i], Nxghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]);
        cudaMemcpyAsync(mybuf->d_xgr[i], xghost_r[i], Nxghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]);
        // y
        cudaMemcpyAsync(mybuf->d_ygl[i], yghost_l[i], Nyghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]);
        cudaMemcpyAsync(mybuf->d_ygr[i], yghost_r[i], Nyghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]);
    }
    GPU::profiler.pop_stopCUDA();

    GPU::h2d_3DArray(src, nslices, gbuf_id, chunk_id);
#endif

}


/* void GPU::upload_xy_ghosts(const uint_t Nxghost, const real_vector_t& xghost_l, const real_vector_t& xghost_r, */
/*         const uint_t Nyghost, const real_vector_t& yghost_l, const real_vector_t& yghost_r, */
/*         const uint_t gbuf_id, const int chunk_id) */
/* { */
/* #ifndef _MUTE_GPU_ */
/*     assert(gbuf_id < _NUM_GPU_BUF_); */

/*     const uint_t s_id = chunk_id % _NUM_STREAMS_; */
/*     GPU_COMM * const mybuf = &gpu_comm[gbuf_id]; */

/*     char prof_item[256]; */
/*     sprintf(prof_item, "SEND GHOSTS (%d)", s_id); */

/*     // TODO: use larger arrays for ghosts to minimize API overhead + */
/*     // increase BW performance. (LOW PRIORITY) */
/*     GPU::profiler.push_startCUDA(prof_item, &stream[s_id]); */
/*     for (int i = 0; i < VSIZE; ++i) */
/*     { */
/*         // x */
/*         cudaMemcpyAsync(mybuf->d_xgl[i], xghost_l[i], Nxghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]); */
/*         cudaMemcpyAsync(mybuf->d_xgr[i], xghost_r[i], Nxghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]); */
/*         // y */
/*         cudaMemcpyAsync(mybuf->d_ygl[i], yghost_l[i], Nyghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]); */
/*         cudaMemcpyAsync(mybuf->d_ygr[i], yghost_r[i], Nyghost*sizeof(Real), cudaMemcpyHostToDevice, stream[s_id]); */
/*     } */
/*     GPU::profiler.pop_stopCUDA(); */
/* #endif */
/* } */


void GPU::h2d_3DArray(const real_vector_t& src, const uint_t nslices,
        const uint_t gbuf_id, const int chunk_id)
{
#ifndef _MUTE_GPU_
    assert(gbuf_id < _NUM_GPU_BUF_);

    const uint_t s_id = chunk_id % _NUM_STREAMS_;
    assert(s_id < _NUM_STREAMS_);
    GPU_COMM * const mybuf = &gpu_comm[gbuf_id];

    char prof_item[256];
    sprintf(prof_item, "SEND 3DARRAY (%d)", s_id);

    GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
    for (int i = 0; i < VSIZE; ++i)
        _h2d_3DArray(mybuf->d_GPUin[i], src[i], nslices, s_id);
    GPU::profiler.pop_stopCUDA();

    cudaEventRecord(event_h2d[s_id], stream[s_id]);
#endif
}


void GPU::d2h_divF(real_vector_t& dst, const uint_t N,
        const uint_t gbuf_id, const int chunk_id)
{
#ifndef _MUTE_GPU_
    assert(gbuf_id < _NUM_GPU_BUF_);

    const uint_t s_id = chunk_id % _NUM_STREAMS_;
    assert(s_id < _NUM_STREAMS_);
    GPU_COMM * const mybuf = &gpu_comm[gbuf_id];

    char prof_item[256];
    sprintf(prof_item, "RECV DIVF (%d)", s_id);

    /* // previous stream has priority, don't interrupt */
    /* const uint_t s_idm1 = ((chunk_id-1) + _NUM_STREAMS_) % _NUM_STREAMS_; */
    /* assert(s_idm1 < _NUM_STREAMS_); */
    /* cudaStreamWaitEvent(stream[s_id], event_d2h[s_idm1]); */

    GPU::profiler.push_startCUDA(prof_item, &stream[s_id]);
    for (int i = 0; i < VSIZE; ++i)
        cudaMemcpyAsync(dst[i], mybuf->d_divF[i], N*sizeof(Real), cudaMemcpyDeviceToHost, stream[s_id]);
    GPU::profiler.pop_stopCUDA();

    cudaEventRecord(event_d2h[s_id], stream[s_id]);
#endif
}


///////////////////////////////////////////////////////////////////////////
// Sync
///////////////////////////////////////////////////////////////////////////
void GPU::wait_h2d(const int chunk_id)
{
#ifndef _MUTE_GPU_
    const uint_t s_id = chunk_id % _NUM_STREAMS_;
    cudaEventSynchronize(event_h2d[s_id]);
#endif
}


void GPU::wait_d2h(const int chunk_id)
{
#ifndef _MUTE_GPU_
    const uint_t s_id = chunk_id % _NUM_STREAMS_;
    cudaEventSynchronize(event_d2h[s_id]);
#endif
}


void GPU::syncGPU()
{
#ifndef _MUTE_GPU_
    cudaDeviceSynchronize();
#endif
}


void GPU::syncStream(const int chunk_id)
{
#ifndef _MUTE_GPU_
    const uint_t s_id = chunk_id % _NUM_STREAMS_;
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
        printf("Can not get memory stats from GPU...\n");
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

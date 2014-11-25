/* File:   bw_bench.cu */
/* Date:   Thu 20 Nov 2014 06:11:40 PM CET */
/* Author: Fabian Wermelinger */
/* Tag:    DDR5 bandwidth bench */
/* Copyright © 2014 Fabian Wermelinger. All Rights Reserved. */
#include <stdio.h>
#include <stdlib.h>

__global__
void DDR5_read(float * const __restrict__ in, float * const __restrict__ out,
        const size_t size, const int nreps)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    int s = tid;
    for (int r=0; r < nreps; ++r)
    {
        const float f0 =  in[(s+0)&(size-1)];
        const float f1 =  in[(s+32768)&(size-1)];
        const float f2 =  in[(s+65536)&(size-1)];
        const float f3 =  in[(s+98304)&(size-1)];
        const float f4 =  in[(s+131072)&(size-1)];
        const float f5 =  in[(s+163840)&(size-1)];
        const float f6 =  in[(s+196608)&(size-1)];
        const float f7 =  in[(s+229376)&(size-1)];
        const float f8 =  in[(s+262144)&(size-1)];
        const float f9 =  in[(s+294912)&(size-1)];
        const float f10 = in[(s+327680)&(size-1)];
        const float f11 = in[(s+360448)&(size-1)];
        const float f12 = in[(s+393216)&(size-1)];
        const float f13 = in[(s+425984)&(size-1)];
        const float f14 = in[(s+458752)&(size-1)];
        const float f15 = in[(s+491520)&(size-1)];
        sum += f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15;
       s = (s+524288)&(size-1);
    }
    out[tid] = sum;
}

__global__
void DDR5_write(float * const __restrict__ out, const size_t size, const int nreps)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const float tidf = (float)tid;
    int s = tid;
    for (int r=0; r < nreps; ++r)
    {
        out[(s+0)&(size-1)]      = tidf;
        out[(s+32768)&(size-1)]  = tidf;
        out[(s+65536)&(size-1)]  = tidf;
        out[(s+98304)&(size-1)]  = tidf;
        out[(s+131072)&(size-1)] = tidf;
        out[(s+163840)&(size-1)] = tidf;
        out[(s+196608)&(size-1)] = tidf;
        out[(s+229376)&(size-1)] = tidf;
        out[(s+262144)&(size-1)] = tidf;
        out[(s+294912)&(size-1)] = tidf;
        out[(s+327680)&(size-1)] = tidf;
        out[(s+360448)&(size-1)] = tidf;
        out[(s+393216)&(size-1)] = tidf;
        out[(s+425984)&(size-1)] = tidf;
        out[(s+458752)&(size-1)] = tidf;
        out[(s+491520)&(size-1)] = tidf;
        s = (s+524288)&(size-1);
    }
}


int main(int argc, char *argv[])
{
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device ID: %d\n", dev);
    printf("Device: %s\n", prop.name);

    size_t threads;
    if (argc == 2)
        threads = atoi(argv[1]);
    else
        threads = 128;
    const size_t N = 32768;
    const size_t blocks = N / threads;
    const size_t nreps = 1024;

    const size_t bytes = 64*1024*1024;
    const size_t nelem = bytes / sizeof(float);
    float *h_in = (float *) malloc(bytes);
    float *h_out= (float *) malloc(bytes);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    srand48(42);
    for (int i = 0; i < nelem; ++i)
        h_in[i] = (float)(drand48()*nelem);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // READ
    DDR5_read<<<512, 64>>>(d_in, d_out, nelem, nreps); // warmup

    cudaEventRecord(start);
    DDR5_read<<<blocks, threads>>>(d_in, d_out, nelem, nreps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    const float ts = elapsed_ms * 1.0e-3;

    const double eff_BW = (double)N * 16.0 * nreps * sizeof(float) / (ts * 1024.*1024.*1024.); // GB/s

    printf("READ:\n");
    printf("Effective Bandwidth = %f GB/s\n\n", eff_BW);


    return 0;
}

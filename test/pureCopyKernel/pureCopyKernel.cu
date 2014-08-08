/* File        : pureCopyKernel.cu */
/* Maintainer  : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Fri 08 Aug 2014 03:39:40 PM CEST */
/* Modified    : Fri 08 Aug 2014 07:00:34 PM CEST */
/* Description : */
#include <stdio.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define NUMREPS 100

__global__
void copy_kernel(float * const __restrict__ in, float * const __restrict__ out,
        const int width, const int heigth, const int nreps)
{
    const int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    const int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    const int index = xIndex + width * yIndex;

    for (int r = 0; r < nreps; ++r)
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            out[index+i*width] = in[index+i*width];
}


int main()
{
    const int size_x = 2048;
    const int size_y = 2048;

    dim3 grid(size_x/TILE_DIM, size_y/TILE_DIM), threads(TILE_DIM, BLOCK_ROWS);

    const int bytes = size_x * size_y * sizeof(float);

    float *h_in = (float *) malloc(bytes);
    float *h_out= (float *) malloc(bytes);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    for (int i = 0; i < (size_x*size_y); ++i)
        h_in[i] = (float)i;

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm up
    copy_kernel<<<grid, threads>>>(d_in, d_out, size_x, size_y, 1);

    cudaEventRecord(start);
    copy_kernel<<<grid, threads>>>(d_in, d_out, size_x, size_y, NUMREPS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_inner;
    cudaEventElapsedTime(&elapsed_inner, start, stop);

    cudaEventRecord(start);
    for ( int i = 0; i < NUMREPS; ++i)
        copy_kernel<<<grid, threads>>>(d_in, d_out, size_x, size_y, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_outer;
    cudaEventElapsedTime(&elapsed_outer, start, stop);

    const float kernel_overhead = (elapsed_outer - elapsed_inner)/NUMREPS;
    const float inner_time = elapsed_inner/NUMREPS;
    const float outer_time = elapsed_outer/NUMREPS;
    const float pure_kernel_time = inner_time - kernel_overhead;

    const float eff_inner_BW = 2.*1000.*bytes/(1024.*1024.*1024.*inner_time); // GB/s
    const float eff_outer_BW = 2.*1000.*bytes/(1024.*1024.*1024.*outer_time); // GB/s
    const float eff_noOverhead_BW = 2.*1000.*bytes/(1024.*1024.*1024.*pure_kernel_time); // GB/s

    printf("REPS INSIDE KERNEL:\n");
    printf("Walltime = %fms\nAvg Kernel Time = %fms\n", elapsed_inner, inner_time);
    printf("Effective Bandwidth = %f GB/s\n\n", eff_inner_BW);

    printf("REPS OUTSIDE KERNEL:\n");
    printf("Walltime = %fms\nAvg Kernel Time = %fms\n", elapsed_outer, outer_time);
    printf("Effective Bandwidth = %f GB/s\n", eff_outer_BW);

    /* printf("\nEffective Bandwith (w/o Overhead) = %f GB/s\n", eff_noOverhead_BW); */

    return 0;
}

/* *
 * PP check
 * */
#include <stdio.h>
#include <cstdlib>
#include "CUDA_Timer.cuh"
using namespace std;

#define SINGLE_PRECISION
#define K20  0
#define K20X 1
#ifndef CARD
#define CARD K20
#endif

#define NLOOP (1<<16)
#define BLOCKS 8192
#define THREADS 128
/* #define BLOCKS 1024 */
/* #define THREADS 64 */

#define FLOP 8

#ifdef SINGLE_PRECISION
typedef float Real;
#define PK20 (0.706 * (13*192) * 2)
#define PK20X (0.732 * (14*192) * 2)
#else
typedef double Real;
#define PK20 (0.706 * (13*64) * 2)
#define PK20X (0.732 * (14*64) * 2)
#endif

#if CARD == K20
#define PMAX PK20
#elif CARD == K20X
#define PMAX PK20X
#else
#error "Processor is not defined"
#endif


__global__
void fma_kernel(Real * const store)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    register Real x1=0.00006f, x2=0.00003f, x3=0.00001f, x4=0.00005f;
    register Real y1=0.01f, y2=0.02f, y3=0.03f, y4=0.04f;

#pragma unroll 256
    for (unsigned int i = 0; i < NLOOP; ++i)
    {
        y1 = x1 + y1*x2;
        y2 = x2 + y2*x3;
        y3 = x3 + y3*x1;
        y4 = x4 + y4*x2;
    }

    store[tid] = y1+y2+y3+y4;
}



int main(const int argc, const char **argv)
{
    Real *dA;
    cudaMalloc((void **)&dA, BLOCKS*THREADS*sizeof(Real));

    GPUtimer timer;
    timer.start();
    fma_kernel<<<BLOCKS, THREADS>>>(dA);
    timer.stop();

    cudaDeviceSynchronize();
    cudaFree(dA);

    const double kTime = timer.elapsed() * 1.0e-3;
    const double Gflop = (FLOP * (double)NLOOP) * double(BLOCKS * THREADS) * 1.0e-9;
    const double Perf  = Gflop / kTime; // Gflops
    const double frac  = Perf/PMAX;
    printf("Performance: %f Gflops (%4.1f%% of Peak)\n", Perf, frac*100);
    printf("Kernel Time: %f s\n", kTime);

    return 0;
}

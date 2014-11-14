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

#define FLOP 6

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
void kernel(Real * const store)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    register Real x1=0.6f, x2=0.6f, x3=0.6f;
    register Real y1=0.1f, y2=0.2f, y3=0.3f;

    register Real x11=0.5f, x22=0.5f, x33=0.5f;
    register Real y11=0.0f, y22=0.0f, y33=0.0f;

    register Real x44=0.2f, x55=0.3f, x66=0.5f;
    register Real y44=0.0f, y55=0.0f, y66=0.0f;

#pragma unroll 256
    for (unsigned int i = 0; i < NLOOP; ++i)
    {
        y1 = x1 + y1*x1;
        y2 = x2 + y2*x2;
        y3 = x3 + y3*x3;

        /* y11 = x11 + y11*x11; */
        /* y22 = x22 + y22*x22; */
        /* y33 = x33 + y33*x33; */

        /* y44 = x44 + y44*x44; */
        /* y55 = x55 + y55*x55; */
        /* y66 = x66 + y66*x66; */
    }

    store[tid] = y1+y2+y3+y11+y22+y33+y44+y55+y66;
}



int main(const int argc, const char **argv)
{
    Real *dA;
    cudaMalloc((void **)&dA, BLOCKS*THREADS*sizeof(Real));

    GPUtimer timer;
    timer.start();
    kernel<<<BLOCKS, THREADS>>>(dA);
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

/* *
 * PP check
 * */
#include <stdio.h>
#include <cstdlib>
#include "CUDA_Timer.cuh"
using namespace std;

#define SINGLE_PRECISION
#define WORK4
/* #define WORK8 */
#define K20  0
#define K20X 1
#ifndef CARD
#define CARD K20
#endif

#define NLOOP (1<<20)
#define BLOCKS 4096
#define THREADS1 256
#define NELEMENTS BLOCKS*THREADS1
#define BYTES NELEMENTS*sizeof(Real)

#if defined(WORK4)
#define THREADS (THREADS1/4) // each thread does 4x more work
#define FLOP 8 // 4 FMA = 8 FLOP
#elif defined(WORK8)
#define THREADS (THREADS1/8) // each thread does 8x more work
#define FLOP 16 // 8 FMA = 16 FLOP
#else
#define THREADS THREADS1
#define FLOP 2
#endif

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
void kernel(Real * const A)
{
    /* const int tid = blockIdx.x * blockDim.x + threadIdx.x; */

#if defined(WORK4)

    // do 4x more work per thread
    register Real x1=1.0f, x2=1.0f, x3=1.0f, x4=1.0f;
    register Real y1, y2, y3, y4;
    /* y1 = A[0*THREADS + tid]; */
    /* y2 = A[1*THREADS + tid]; */
    /* y3 = A[2*THREADS + tid]; */
    /* y4 = A[3*THREADS + tid]; */

#pragma unroll 128
    for (unsigned int i = 0; i < NLOOP; ++i)
    {
        y1 = x1 + 0.500002319803f*x2 + 0.100001450087f*x3;
        y2 = x2 + 0.500002319803f*x1 + 0.100001450087f*x4;
        y3 = x3 + 0.100001450087f*x1 + 0.500002319803f*x4;
        y4 = x4 + 0.100001450087f*x2 + 0.500002319803f*x3;

        x1 = y1 + 0.500002319803f*y2 + 0.100001450087f*y3;
        x2 = y2 + 0.500002319803f*y1 + 0.100001450087f*y4;
        x3 = y3 + 0.100001450087f*y1 + 0.500002319803f*y4;
        x4 = y4 + 0.100001450087f*y2 + 0.500002319803f*y3;
    }

    /* A[0*THREADS + tid] = e; */
    /* A[1*THREADS + tid] = f; */
    /* A[2*THREADS + tid] = g; */
    /* A[3*THREADS + tid] = h; */

#elif defined(WORK8)

    // do 8x more work per thread
    register Real a, b, c, d, e, f, g, h;
    register Real q=0, r=0, s=0, t=0, u=0, v=0, w=0, x=0;
    a = A[0*THREADS + tid];
    b = A[1*THREADS + tid];
    c = A[2*THREADS + tid];
    d = A[3*THREADS + tid];
    e = A[4*THREADS + tid];
    f = A[5*THREADS + tid];
    g = A[6*THREADS + tid];
    h = A[7*THREADS + tid];

#pragma unroll 64
    for (unsigned int i = 0; i < NLOOP; ++i)
    {
        q = a + q * a;
        r = b + r * b;
        s = c + s * c;
        t = d + t * d;
        u = e + u * e;
        v = f + v * f;
        w = g + w * g;
        x = h + x * h;
    }

    A[0*THREADS + tid] = q;
    A[1*THREADS + tid] = r;
    A[2*THREADS + tid] = s;
    A[3*THREADS + tid] = t;
    A[4*THREADS + tid] = u;
    A[5*THREADS + tid] = v;
    A[6*THREADS + tid] = w;
    A[7*THREADS + tid] = x;

#else

    // minimal work per thread
    register Real a, b=0;
    a = A[tid];

#pragma unroll 128
    for (unsigned int i = 0; i < NLOOP; ++i)
        b = a + b * a;

    A[tid] = b;

#endif
}



int main(const int argc, const char **argv)
{
    Real *dA, *A;
    cudaMalloc( (void**)&dA, BYTES );
    A = (Real *)malloc(BYTES);
    for (int i = 0; i < NELEMENTS; ++i)
        A[i] = 1.0;
    cudaMemcpy(dA, A, BYTES, cudaMemcpyHostToDevice);

    GPUtimer timer;
    timer.start();
    kernel<<<BLOCKS, THREADS>>>(dA);
    timer.stop();

    cudaDeviceSynchronize();

    cudaFree(dA);
    free(A);

    const double Gflop = (FLOP * (double)NLOOP) * double(BLOCKS * THREADS) * 1.0e-9;
    const double Perf  = Gflop / (timer.elapsed() * 1.0e-3); // Gflops
    const double frac  = Perf/PMAX;
    printf("Performance: %f Gflops (%4.1f%% of Peak)\n", Perf, frac*100);

    return 0;
}

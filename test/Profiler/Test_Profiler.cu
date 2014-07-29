/* File        : Test_Profiler.cu */
/* Maintainer  : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Tue 29 Jul 2014 02:13:51 PM CEST */
/* Modified    : Tue 29 Jul 2014 06:09:33 PM CEST */
/* Description : */
#include "Profiler.h"


__global__
void kernel(float *c)
{
    float a=0, b=1;
    for (unsigned long int i = 0; i < 100000000; ++i)
        a = a*b + b;
    *c = a;
}


int main()
{
    float *d;
    cudaMalloc((void **)&d, sizeof(float));

    Profiler prof;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < 5; ++i)
    {
        prof.push_startCUDA("My CUDA Kernel", &stream);
        kernel<<<1,1, 0, stream>>>(d);
        prof.pop_stopCUDA();
    }

    cudaStreamDestroy(stream);


    prof.printSummary();
}

/* File        : Profiler.cu */
/* Maintainer  : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Tue 29 Jul 2014 01:56:20 PM CEST */
/* Modified    : Tue 29 Jul 2014 05:25:51 PM CEST */
/* Description : CUDA profiling tools */
#include "Profiler.h"
#include <sys/time.h>
#include <stdio.h>

//#include <tbb/tick_count.h>
//using namespace tbb;

void ProfileAgent::_getTime(ClockTime& time)
{
	//time = tick_count::now();
	gettimeofday(&time, NULL);
}

double ProfileAgent::_getElapsedTime(const ClockTime& tS, const ClockTime& tE)
{
	return (tE.tv_sec - tS.tv_sec) + 1e-6 * (tE.tv_usec - tS.tv_usec);
	//return (tE - tS).seconds();
}

void ProfileAgentCUDA::_getTime(const void *event, const void *stream)
{
    if (stream)
        cudaEventRecord(*(cudaEvent_t *)event, *(cudaStream_t *)stream);
    else
        cudaEventRecord(*(cudaEvent_t *)event, 0);
}

double ProfileAgentCUDA::_getElapsedTime(const void *tStart, const void *tEnd)
{
    cudaEventSynchronize(*(cudaEvent_t *)tEnd);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, *(cudaEvent_t *)tStart, *(cudaEvent_t *)tEnd);
    return (double)elapsed*1.0e-3;
}

void ProfileAgentCUDA::_createEvent(void **event)
{
    if (*event) _destroyEvent(*(cudaEvent_t **)event);
    cudaEvent_t *heap_event = new cudaEvent_t;
    cudaEventCreate(heap_event);
    *(cudaEvent_t **)event = heap_event;
}

void ProfileAgentCUDA::_destroyEvent(void *event)
{
    if (event)
    {
        cudaEventDestroy(*(cudaEvent_t *)event);
        delete (cudaEvent_t *)event;
    }
}

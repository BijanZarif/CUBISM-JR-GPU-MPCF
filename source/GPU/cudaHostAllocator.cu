/* *
 * cudaHostAllocator.cu
 *
 * Created by Fabian Wermelinger on 06/06/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <cstdlib>

void *_cudaAllocHost(const std::size_t bytes)
{
    void *palloc;
    cudaHostAlloc(&palloc, bytes, cudaHostAllocDefault);
    return palloc;
}

void _cudaFreeHost(void *ptr)
{
    cudaFreeHost(ptr);
}

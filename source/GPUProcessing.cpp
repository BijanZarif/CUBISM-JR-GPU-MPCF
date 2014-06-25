/* *
 * GPUProcessing.cpp
 *
 * Created by Fabian Wermelinger on 5/28/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "GPUProcessing.h"

#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>
#include <stdio.h>
using namespace std;

#define ID3(x,y,z,NX,NY) ((x) + (NX) * ((y) + (NY) * (z)))


GPUProcessing::GPUProcessing(const uint_t BSX, const uint_t BSY, const uint_t BSZ, const uint_t CL)
    :
    BSX_GPU(BSX), BSY_GPU(BSY), BSZ_GPU(BSZ), CHUNK_LENGTH(CL),
    SLICE_GPU( BSX * BSY ), REM( BSZ % CL ),
    N_chunks( (BSZ + CL - 1) / CL ),
    GPU_input_size( SLICE_GPU * (CL+6) ),
    GPU_output_size( SLICE_GPU * CL ),
    N_xyghosts(3*BSY*CL), // WARNING: assumes cubic domain!
    N_zghosts(3*SLICE_GPU), // WARNING: assumes cubic domain!
    BUFFER1(GPU_input_size, GPU_output_size, N_xyghosts, N_zghosts),
    BUFFER2(GPU_input_size, GPU_output_size, N_xyghosts, N_zghosts)
{
    if (0 < REM && REM < 3)
    {
        cerr << "[GPUProcessing ERROR: Too few slices in the last chunk (have " << REM << " slices, should be 3 or higher)" << endl;
        exit(1);
    }

    chatty = QUIET;

    _alloc_GPU();

    _reset();

    buffer          = &BUFFER1;
    previous_buffer = &BUFFER2;

    previous_length   = (1 - !REM)*REM + (!REM)*CHUNK_LENGTH;
    previous_iz       = (N_chunks-1) * CHUNK_LENGTH;
    previous_chunk_id = N_chunks;
}


GPUProcessing::~GPUProcessing()
{
    _free_GPU();
}


void GPUProcessing::_alloc_GPU()
{
    // allocate GPU memory
    GPU::alloc((void**) &maxSOS, BSX_GPU, BSY_GPU, BSZ_GPU, CHUNK_LENGTH);
    gpu_allocation = ALLOCATED;
}


void GPUProcessing::_free_GPU()
{
    GPU::dealloc();
    gpu_allocation = FREE;
}


void GPUProcessing::_reset()
{
    if (N_chunks == 1)
    {
        // whole chunk fits on the GPU
        chunk_state = SINGLE;
        current_length = BSZ_GPU;
    }
    else
    {
        chunk_state = FIRST;
        current_length = CHUNK_LENGTH;
    }
    current_iz = 0;
    current_chunk_id = 1;
}


void GPUProcessing::_copy_AOS_to_SOA(RealPtrVec_t& dst, const Real * const src, const uint_t gptfloats, const uint_t Nelements)
{
    Real * const ptr[7] = {dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6]};

#pragma omp parallel for
    for (int i = 0; i < Nelements; ++i)
        for (int comp = 0; comp < 7; ++comp)
            (ptr[comp])[i] = src[i*gptfloats + comp];
}


void GPUProcessing::_copy_SOA_to_AOS(Real * const dst, const RealPtrVec_t& src, const uint_t gptfloats, const uint_t Nelements)
{
    const Real * const ptr[7] = {src[0], src[1], src[2], src[3], src[4], src[5], src[6]};

#pragma omp parallel for
    for (int i = 0; i < Nelements; ++i)
        for (int comp = 0; comp < 7; ++comp)
            dst[i*gptfloats + comp] = (ptr[comp])[i];
}


void GPUProcessing::_init_next_subdomain()
{
    previous_length   = current_length;
    previous_iz       = current_iz;
    previous_chunk_id = current_chunk_id;

    current_iz     += current_length;
    ++current_chunk_id;

    if (current_chunk_id > N_chunks)
        _reset();
    else if (current_chunk_id == N_chunks)
    {
        current_length  = (1 - !REM)*REM + (!REM)*CHUNK_LENGTH;
        chunk_state     = LAST;
    }
    else
    {
        current_length = CHUNK_LENGTH;
        chunk_state    = INTERMEDIATE;
    }

    // use a new host buffer
    _switch_buffer();
}


void GPUProcessing::_printSOA(const Real * const in)
{
    for (int iz = 0; iz < current_length+6; ++iz)
    {
        for (int iy = 0; iy < BSY_GPU; ++iy)
        {
            for (int ix = 0; ix < BSX_GPU; ++ix)
                cout << in[ID3(ix, iy, iz, BSX_GPU, BSY_GPU)] << '\t';
            cout << endl;
        }
        cout << endl;
    }
}


void GPUProcessing::_info_current_chunk()
{
    string state;
    switch (chunk_state)
    {
        case FIRST:        state = "FIRST"; break;
        case INTERMEDIATE: state = "INTERMEDIATE"; break;
        case LAST:         state = "LAST"; break;
        case SINGLE:       state = "SINGLE"; break;
    }
    printf("[CURRENT CHUNK:        \t%d/%d]\n", current_chunk_id, N_chunks);
    printf("[CURRENT CHUNK STATE:  \t%s]\n", state.c_str());
    printf("[CURRENT CHUNK LENGTH: \t%d/%d]\n", current_length, CHUNK_LENGTH);
    printf("[PREVIOUS CHUNK LENGTH:\t%d/%d]\n", previous_length, CHUNK_LENGTH);
    printf("[CURRENT Z-POS:        \t%d/%d]\n", current_iz, BSZ_GPU);
    printf("[NUMBER OF NODES:      \t%d]\n", SLICE_GPU * current_length);
}

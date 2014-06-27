/* *
 * GPUlab.cpp
 *
 * Created by Fabian Wermelinger on 5/28/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include "GPUlab.h"


GPUlab::GPUlab(GridMPI& G, const uint_t CL)
    :
        grid(G), CHUNK_LENGTH(CL), REM( sizeZ % CL ),
        N_chunks( (sizeZ + CL - 1) / CL ),
        GPU_input_size( SLICE_GPU * (CL+6) ),
        GPU_output_size( SLICE_GPU * CL ),
        cart_world(grid.getCartComm()),
        request(6), status(6),
        halox(3*sizeY*sizeZ), // all domain
        haloy(sizeX*3*sizeZ), // all domain
        haloz(sizeX*sizeY*3), // all domain
        BUFFER1(GPU_input_size, GPU_output_size, 3*sizeY*CL, sizeX*3*CL), // per chunk
        BUFFER2(GPU_input_size, GPU_output_size, 3*sizeY*CL, sizeX*3*CL)  // per chunk
{
    if (REM != 0) // can be solved later
    {
        fprintf(stderr, "[GPUlab ERROR: CURRENTLY CHUNK LENGTHS MUST BE A MULTIPLE of GridMPI::sizeZ\n");
        exit(1);
    }

    if (0 < REM && REM < 3)
    {
        fprintf(stderr, "[GPUlab ERROR: Too few slices in the last chunk (have %d slices, should be 3 or higher)\n", REM);
        exit(1);
    }

    chatty = QUIET;

    _alloc_GPU();

    _reset();

    buffer          = &BUFFER1; // are swapped with _swap_buffer()
    previous_buffer = &BUFFER2;

    previous_length   = (1 - !REM)*REM + (!REM)*CHUNK_LENGTH;
    previous_iz       = (N_chunks-1) * CHUNK_LENGTH;
    previous_chunk_id = N_chunks;

    int mycoords[3];
    grid.peindex(mycoords);
    for (int i = 0; i < 3; ++i)
    {
        myFeature[i*2 + 0] = mycoords[i] == 0 ? SKIN : FLESH;
        myFeature[i*2 + 1] = mycoords[i] == grid.getBlocksPerDimension(i)-1 ? SKIN : FLESH;
    }
    grid.getNeighborRanks(nbr);
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE
///////////////////////////////////////////////////////////////////////////////
void GPUlab::_show_feature()
{
    int c[3];
    grid.peindex(c);
    for (int i = 0; i < 3; ++i)
    {
        printf("MPI coords (%d,%d,%d) -- i = %d, left:  %s\n", c[0], c[1], c[2], i, myFeature[2*i+0] == SKIN ? "Skin" : "Flesh");
        printf("MPI coords (%d,%d,%d) -- i = %d, right: %s\n", c[0], c[1], c[2], i, myFeature[2*i+1] == SKIN ? "Skin" : "Flesh");
    }
}


void GPUlab::_start_info_current_chunk(const std::string title = "")
{
    std::string state;
    switch (chunk_state)
    {
        case FIRST:        state = "FIRST"; break;
        case INTERMEDIATE: state = "INTERMEDIATE"; break;
        case LAST:         state = "LAST"; break;
        case SINGLE:       state = "SINGLE"; break;
    }
    printf("{\n");
    printf("\t%s\n", title.c_str());
    printf("\t[CURRENT CHUNK:        \t%d/%d]\n", current_chunk_id, N_chunks);
    printf("\t[CURRENT CHUNK STATE:  \t%s]\n", state.c_str());
    printf("\t[CURRENT CHUNK LENGTH: \t%d/%d]\n", current_length, CHUNK_LENGTH);
    printf("\t[PREVIOUS CHUNK LENGTH:\t%d/%d]\n", previous_length, CHUNK_LENGTH);
    printf("\t[CURRENT Z-POS:        \t%d/%d]\n", current_iz, sizeZ);
    printf("\t[NUMBER OF NODES:      \t%d]\n", SLICE_GPU * current_length);
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC
///////////////////////////////////////////////////////////////////////////////
void GPUlab::load_ghosts(const double t = 0)
{

    if (myFeature[0] == FLESH) _copysend_halos< halomap_x<0,sizeY,3>        >(0, &halox.send_left[0], halox.Nhalo, 0, 3, 0, sizeY, 0, sizeZ);
    if (myFeature[1] == FLESH) _copysend_halos< halomap_x<-sizeX+3,sizeY,3> >(1, &halox.send_right[0],halox.Nhalo, sizeX-3, sizeX, 0, sizeY, 0, sizeZ);
    if (myFeature[2] == FLESH) _copysend_halos< halomap_y<0,sizeX,3>        >(2, &haloy.send_left[0], haloy.Nhalo, 0, sizeX, 0, 3, 0, sizeZ);
    if (myFeature[3] == FLESH) _copysend_halos< halomap_y<-sizeY+3,sizeX,3> >(3, &haloy.send_right[0],haloy.Nhalo, 0, sizeX, sizeY-3, sizeY, 0, sizeZ);
    if (myFeature[4] == FLESH) _copysend_halos(4, &haloz.send_left[0], haloz.Nhalo, 0);
    if (myFeature[5] == FLESH) _copysend_halos(5, &haloz.send_right[0],haloz.Nhalo, sizeZ-3);

    if (myFeature[0] == FLESH) _issue_recv(&halox.recv_left[0], halox.Allhalos, 0); // x/yhalos directly into pinned mem and H2D ALLLLLLL
    if (myFeature[1] == FLESH) _issue_recv(&halox.recv_right[0], halox.Allhalos, 1);
    if (myFeature[2] == FLESH) _issue_recv(&haloy.recv_left[0], haloy.Allhalos, 2);
    if (myFeature[3] == FLESH) _issue_recv(&haloy.recv_right[0], haloy.Allhalos, 3);
    if (myFeature[4] == FLESH) _issue_recv(&haloz.recv_left[0], haloz.Allhalos, 4); // receive into buffer->zghost_l ? why not
    if (myFeature[5] == FLESH) _issue_recv(&haloz.recv_right[0], haloz.Allhalos, 5);

    // TEST
    /* _show_feature(); */
    /* const Real * const out = &halox.recv_right[0]; */
    /* for (int iz = 0; iz < 3; ++iz) */
    /* { */
    /*     for (int iy = 0; iy < sizeY; ++iy) */
    /*     { */
    /*         for (int ix = 0; ix < sizeX; ++ix) */
    /*         { */
    /*             printf("%f\t", out[ix+sizeX*(iy+sizeY*iz)]); */
    /*         } */
    /*         printf("\n"); */
    /*     } */
    /*     printf("\n"); */
    /* } */
    /* int c[3]; */
    /* grid.peindex(c); */
    /* printf("(%d,%d,%d){x_l = %f, x_r = %f, y_l = %f, y_r = %f, z_l = %f, z_r = %f}\n", */
    /*         c[0], c[1], c[2], */
    /*         halox.recv_left[0], halox.recv_right[0], */
    /*         haloy.recv_left[0], haloy.recv_right[0], */
    /*         haloz.recv_left[0], haloz.recv_right[0]); */

    _apply_bc(t);

    // swap receive buffer / ghosts
}

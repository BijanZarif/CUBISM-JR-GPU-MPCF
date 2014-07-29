/* File        : GPUkernels_dev.cu */
/* Maintainer  : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Tue 29 Jul 2014 11:25:42 AM CEST */
/* Modified    : Tue 29 Jul 2014 11:42:54 AM CEST */
/* Description : Development stuff, which is taken out of the main kernel
 *               source GPUkernels.cu.  This source is not used for any
 *               compilation or use, it merely is a trunk to keep some
 *               experimental kernel source, the main kernel source code
 *               remains clean. */


#define ARBITRARY_SLICE_DIM
#ifdef ARBITRARY_SLICE_DIM
// BUGGY !!!!!!!!!!!!!!! + likely to yield less performance as if slice
// dimensions would be ineger multiples of TILE_DIM.
__global__
void _xextraterm_hllc(const uint_t nslices,
        const Real * const Gm, const Real * const Gp,
        const Real * const Pm, const Real * const Pp,
        const Real * const vel,
        Real * const sumG, Real * const sumP, Real * const divU)
{
    /* *
     * Computes x-contribution for the right hand side of the advection
     * equations.  Maps two values on cell faces to one value at the cell
     * center.  NOTE: The assignment here is "="
     * */
    uint_t ix = blockIdx.x * TILE_DIM + threadIdx.x;
    uint_t iy = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ Real smem[TILE_DIM*(TILE_DIM+1)];

    if (ix < NX && iy < NY)
    {
        // compute this blocks actual stride for shared memory access.  This is
        // required for arbitray dimensions. Stride for fastest moving index
        // must be stride+1 to avoid bank conflicts.
        const uint_t NXsmem = (NX < TILE_DIM*(blockIdx.x+1)) ? (NX - blockIdx.x*TILE_DIM + 1) : (TILE_DIM+1);
        const uint_t NYsmem = (NY < TILE_DIM*(blockIdx.y+1)) ? (NY - blockIdx.y*TILE_DIM) : TILE_DIM;

        // transpose
        const uint_t iyT = blockIdx.y * TILE_DIM + threadIdx.x;
        const uint_t ixT = blockIdx.x * TILE_DIM + threadIdx.y;

        for (uint_t iz = 0; iz < nslices; ++iz)
        {
            // G
            smem[threadIdx.x*NYsmem + threadIdx.y] = Gp[ID3(iyT,ixT,iz,NY,NXP1)] + Gm[ID3(iyT,(ixT+1),iz,NY,NXP1)];
            __syncthreads();
            sumG[ID3(ix,iy,iz,NX,NY)] = smem[threadIdx.y*NXsmem + threadIdx.x];
            __syncthreads();

            // P
            smem[threadIdx.x*NYsmem + threadIdx.y] = Pp[ID3(iyT,ixT,iz,NY,NXP1)] + Pm[ID3(iyT,(ixT+1),iz,NY,NXP1)];
            __syncthreads();
            sumP[ID3(ix,iy,iz,NX,NY)] = smem[threadIdx.y*NXsmem + threadIdx.x];
            __syncthreads();

            // Velocity on cell faces
            smem[threadIdx.x*NYsmem + threadIdx.y] = vel[ID3(iyT,(ixT+1),iz,NY,NXP1)] - vel[ID3(iyT,ixT,iz,NY,NXP1)];
            __syncthreads();
            divU[ID3(ix,iy,iz,NX,NY)] = smem[threadIdx.y*NXsmem + threadIdx.x];
            __syncthreads();
        }
    }
}
#endif

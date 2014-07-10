// GPU test suite

#include <stdio.h>
#include <vector>
using namespace std;

#include "GPU.h"

// tests
#include "TestGPUKernel.h"


int main(int argc, const char *argv[])
{
    int *dummy;
    const uint_t NX = NodeBlock::sizeX;
    const uint_t NY = NodeBlock::sizeY;
    const uint_t NZ = NodeBlock::sizeZ;
    const uint_t CW = NZ;

    // create life
    GPU::alloc((void**)&dummy, NX, NY, NZ, CW);

    // make some xyghosts
    const uint_t Nxghosts = 3 * NY * CW;
    const uint_t Nyghosts = NX * 3 * CW;
    cuda_vector_t xg_L(7*Nxghosts);
    cuda_vector_t xg_R(7*Nxghosts);
    cuda_vector_t yg_L(7*Nyghosts);
    cuda_vector_t yg_R(7*Nyghosts);
    for (int iz = 0; iz < NZ; ++iz)
        for (int ix = 0; ix < 3; ++ix)
            for (int iy = 0; iy < NY; ++iy)
            {
                /* printf("ghostmap::X(%d,%d,%d) = %d\n",ix,iy,iz,ghostmap::X(ix,iy,iz)); */
                xg_L[0*Nxghosts + ghostmap::X(ix,iy,iz)] = -(Real)ghostmap::X(ix,iy,iz);
                xg_L[1*Nxghosts + ghostmap::X(ix,iy,iz)] = -(Real)ghostmap::X(ix,iy,iz);
                xg_L[2*Nxghosts + ghostmap::X(ix,iy,iz)] = -(Real)ghostmap::X(ix,iy,iz);
                xg_L[3*Nxghosts + ghostmap::X(ix,iy,iz)] = -(Real)ghostmap::X(ix,iy,iz);
                xg_L[4*Nxghosts + ghostmap::X(ix,iy,iz)] = -(Real)ghostmap::X(ix,iy,iz);
                xg_L[5*Nxghosts + ghostmap::X(ix,iy,iz)] = -(Real)ghostmap::X(ix,iy,iz);
                xg_L[6*Nxghosts + ghostmap::X(ix,iy,iz)] = -(Real)ghostmap::X(ix,iy,iz);

                xg_R[0*Nxghosts + ghostmap::X(ix,iy,iz)] = (Real)ghostmap::X(ix,iy,iz);
                xg_R[1*Nxghosts + ghostmap::X(ix,iy,iz)] = (Real)ghostmap::X(ix,iy,iz);
                xg_R[2*Nxghosts + ghostmap::X(ix,iy,iz)] = (Real)ghostmap::X(ix,iy,iz);
                xg_R[3*Nxghosts + ghostmap::X(ix,iy,iz)] = (Real)ghostmap::X(ix,iy,iz);
                xg_R[4*Nxghosts + ghostmap::X(ix,iy,iz)] = (Real)ghostmap::X(ix,iy,iz);
                xg_R[5*Nxghosts + ghostmap::X(ix,iy,iz)] = (Real)ghostmap::X(ix,iy,iz);
                xg_R[6*Nxghosts + ghostmap::X(ix,iy,iz)] = (Real)ghostmap::X(ix,iy,iz);
            }
    for (int iz = 0; iz < NZ; ++iz)
        for (int iy = 0; iy < 3; ++iy)
            for (int ix = 0; ix < NX; ++ix)
            {
                /* printf("ghostmap::Y(%d,%d,%d) = %d\n",ix,iy,iz,ghostmap::Y(ix,iy,iz)); */
                yg_L[0*Nxghosts + ghostmap::Y(ix,iy,iz)] = -(Real)ghostmap::Y(ix,iy,iz);
                yg_L[1*Nxghosts + ghostmap::Y(ix,iy,iz)] = -(Real)ghostmap::Y(ix,iy,iz);
                yg_L[2*Nxghosts + ghostmap::Y(ix,iy,iz)] = -(Real)ghostmap::Y(ix,iy,iz);
                yg_L[3*Nxghosts + ghostmap::Y(ix,iy,iz)] = -(Real)ghostmap::Y(ix,iy,iz);
                yg_L[4*Nxghosts + ghostmap::Y(ix,iy,iz)] = -(Real)ghostmap::Y(ix,iy,iz);
                yg_L[5*Nxghosts + ghostmap::Y(ix,iy,iz)] = -(Real)ghostmap::Y(ix,iy,iz);
                yg_L[6*Nxghosts + ghostmap::Y(ix,iy,iz)] = -(Real)ghostmap::Y(ix,iy,iz);

                yg_R[0*Nxghosts + ghostmap::Y(ix,iy,iz)] = (Real)ghostmap::Y(ix,iy,iz);
                yg_R[1*Nxghosts + ghostmap::Y(ix,iy,iz)] = (Real)ghostmap::Y(ix,iy,iz);
                yg_R[2*Nxghosts + ghostmap::Y(ix,iy,iz)] = (Real)ghostmap::Y(ix,iy,iz);
                yg_R[3*Nxghosts + ghostmap::Y(ix,iy,iz)] = (Real)ghostmap::Y(ix,iy,iz);
                yg_R[4*Nxghosts + ghostmap::Y(ix,iy,iz)] = (Real)ghostmap::Y(ix,iy,iz);
                yg_R[5*Nxghosts + ghostmap::Y(ix,iy,iz)] = (Real)ghostmap::Y(ix,iy,iz);
                yg_R[6*Nxghosts + ghostmap::Y(ix,iy,iz)] = (Real)ghostmap::Y(ix,iy,iz);
            }
    RealPtrVec_t xghost_l(7), xghost_r(7), yghost_l(7), yghost_r(7);
    for (int i = 0; i < 7; ++i)
    {
        xghost_l[i] = &xg_L[i*Nxghosts];
        xghost_r[i] = &xg_R[i*Nxghosts];
        yghost_l[i] = &yg_L[i*Nyghosts];
        yghost_r[i] = &yg_R[i*Nyghosts];
    }
    GPU::upload_xy_ghosts(Nxghosts, xghost_l, xghost_r, Nyghosts, yghost_l, yghost_r);
    GPU::syncGPU();

    // make some interior
    const uint_t Ninput = NX * NY * (NZ+6);
    cuda_vector_t interior(7*Ninput);
    for (int iz = 0; iz < NZ+6; ++iz)
        for (int iy = 0; iy < NY; ++iy)
            for (int ix = 0; ix < NX; ++ix)
            {
                const uint_t idx = ID3(ix,iy,iz,NX,NY);
                interior[0*Ninput + idx] = (Real)idx;
                interior[1*Ninput + idx] = (Real)idx;
                interior[2*Ninput + idx] = (Real)idx;
                interior[3*Ninput + idx] = (Real)idx;
                interior[4*Ninput + idx] = (Real)idx;
                interior[5*Ninput + idx] = (Real)idx;
                interior[6*Ninput + idx] = (Real)idx;
            }
    RealPtrVec_t in3D(7);
    for (int i = 0; i < 7; ++i)
        in3D[i] = &interior[i*Ninput];
    GPU::h2d_3DArray(in3D, NX, NY, NZ+6);
    GPU::h2d_3DArray_wait();

    // run Forrest
    TestGPUKernel kernel;
    kernel.run();
    GPU::syncGPU();

    // destroy life
    GPU::dealloc();

    return 0;
}

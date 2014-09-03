/* File        : GPUdebug.h */
/* Creator     : Fabian Wermelinger <fabianw@student.ethz.ch> */
/* Created     : Wed 03 Sep 2014 11:55:22 AM CEST */
/* Modified    : Wed 03 Sep 2014 11:55:56 AM CEST */
/* Description : */
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

void _TEST_dump(const Real * const d_data, const size_t bytes, const string fname = "data.bin")
{
    Real *h_data = (Real *)malloc(bytes);
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    ofstream out(fname.c_str(), std::ofstream::binary);
    out.write((char *)h_data, bytes);
    out.close();
    free(h_data);
}

__global__
void _TEST_CONV(DevicePointer inout,
        DevicePointer xgL, DevicePointer xgR,
        DevicePointer ygL, DevicePointer ygR)
{
    const Real r_ref = 1.5f;
    const Real u_ref = 1.0f;
    const Real v_ref = 1.0f;
    const Real w_ref = 1.0f;
    const Real e_ref = 1.0f;
    const Real G_ref = 2.0f;
    const Real P_ref = 3.0f;

    // test main body
    const uint_t Ninout = NX * NY * (NodeBlock::sizeZ + 6);
    for (int i = 0; i < Ninout; ++i)
    {
        /* printf("%f\n", inout.w[i]); */
        assert(inout.r[i] == r_ref);
        assert(inout.u[i] == u_ref);
        assert(inout.v[i] == v_ref);
        assert(inout.w[i] == w_ref);
        assert(inout.e[i] == e_ref);
        assert(inout.G[i] == G_ref);
        assert(inout.P[i] == P_ref);
    }

    // test xghosts
    const uint_t Nxghost = 3*NY*(NodeBlock::sizeZ);
    for (int i = 0; i < Nxghost; ++i)
    {
        assert(xgR.r[i] == r_ref);
        assert(xgR.u[i] == u_ref);
        assert(xgR.v[i] == v_ref);
        assert(xgR.w[i] == w_ref);
        assert(xgR.e[i] == e_ref);
        assert(xgR.G[i] == G_ref);
        assert(xgR.P[i] == P_ref);

        assert(xgL.r[i] == r_ref);
        assert(xgL.u[i] == u_ref);
        assert(xgL.v[i] == v_ref);
        assert(xgL.w[i] == w_ref);
        assert(xgL.e[i] == e_ref);
        assert(xgL.G[i] == G_ref);
        assert(xgL.P[i] == P_ref);
    }

    // test yghosts
    const uint_t Nyghost = NX*3*(NodeBlock::sizeZ);
    for (int i = 0; i < Nyghost; ++i)
    {
        assert(ygR.r[i] == r_ref);
        assert(ygR.u[i] == u_ref);
        assert(ygR.v[i] == v_ref);
        assert(ygR.w[i] == w_ref);
        assert(ygR.e[i] == e_ref);
        assert(ygR.G[i] == G_ref);
        assert(ygR.P[i] == P_ref);

        assert(ygL.r[i] == r_ref);
        assert(ygL.u[i] == u_ref);
        assert(ygL.v[i] == v_ref);
        assert(ygL.w[i] == w_ref);
        assert(ygL.e[i] == e_ref);
        assert(ygL.G[i] == G_ref);
        assert(ygL.P[i] == P_ref);
    }
}

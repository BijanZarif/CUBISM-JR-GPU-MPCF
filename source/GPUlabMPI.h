/* *
 * GPUlabMPI.h
 *
 * Created by Fabian Wermelinger on 5/28/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#pragma once

#include "GPU.h"
#include "GridMPI.h"
#include "Types.h"
#include "Profiler.h"

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#ifdef _USE_HDF_
#include <hdf5.h>
#endif

#ifdef _FLOAT_PRECISION_
#define _MPI_REAL_ MPI_FLOAT
#else
#define _MPI_REAL_ MPI_DOUBLE
#endif


class GPUlabMPI
{
    public:

        static const uint_t sizeX = GridMPI::sizeX;
        static const uint_t sizeY = GridMPI::sizeY;
        static const uint_t sizeZ = GridMPI::sizeZ;
        static const uint_t SLICE_GPU = GridMPI::sizeX * GridMPI::sizeY;


    private:

        enum {FIRST, INTERMEDIATE, LAST, SINGLE} chunk_state;
        enum {ALLOCATED, FREE} gpu_allocation;
        enum {QUIET=0, VERBOSE} chatty;

        const bool isroot;

        // GPU BUFFER SIZES
        const uint_t GPU_input_size;
        const uint_t GPU_output_size;

        // CHUNK RELATED
        const uint_t nslices;
        const uint_t nslices_last; // nslices_last = sizeZ % nslices
        const uint_t nchunks;

        Profiler *profiler;

        int* maxSOS; // pointer to mapped memory CPU/GPU

        ///////////////////////////////////////////////////////////////////////
        // HALOS / MPI COMMUNICATION
        ///////////////////////////////////////////////////////////////////////
        const MPI_Comm cart_world;
        std::vector<MPI_Request> request;
        std::vector<MPI_Status> status;

        int nbr[6]; // neighbor ranks

        struct Halo // hello halo
        {
            static const uint_t NVAR = GridMPI::NVAR; // number of variables in set
            const uint_t Nhalo;
            const uint_t Allhalos;
            std::vector<Real> send_left, send_right; // for position x1 < x2, then x1 = buf_left, x2 = buf_right
            std::vector<Real> recv_left, recv_right;
            real_vector_t left, right;
            Halo(const uint_t sizeHalo) :
                Nhalo(sizeHalo), Allhalos(NVAR*sizeHalo),
                send_left(NVAR*sizeHalo, 0.0),
                send_right(NVAR*sizeHalo, 0.0),
                recv_left(NVAR*sizeHalo, 0.0),  left(NVAR, NULL),
                recv_right(NVAR*sizeHalo, 0.0), right(NVAR, NULL)
            {
                for (int i = 0; i < NVAR; ++i)
                {
                    // for convenience
                    left[i]  = &recv_left [i * Nhalo];
                    right[i] = &recv_right[i * Nhalo];
                }
            }
        };

        // MPI
        inline void _issue_send(const Real * const sendbuf, const uint_t Nelements, const uint_t sender)
        {
            // why is the send buffer not a const pointer?? 3.0 Standard says
            // different
            MPI_Isend(const_cast<Real * const>(sendbuf), Nelements, _MPI_REAL_, nbr[sender], sender, cart_world, &request[sender]);
        }

        inline void _issue_recv(Real * const recvbuf, const uint_t Nelements, const uint_t receiver)
        {
            MPI_Recv(recvbuf, Nelements, _MPI_REAL_, nbr[receiver], MPI_ANY_TAG, cart_world, &status[receiver]);
        }

        // Halo extraction
        template <index_map map>
        void _copysend_halos(const int sender, Real * const cpybuf, const uint_t Nhalos, const int xS, const int xE, const int yS, const int yE, const int zS, const int zE);
        void _copysend_halos(const int sender, Real * const cpybuf, const uint_t Nhalos, const int zS);


        ///////////////////////////////////////////////////////////////////////
        // HOST BUFFERS (pinned)
        ///////////////////////////////////////////////////////////////////////
        struct HostBuffer
        {
            static const uint_t NVAR = GridMPI::NVAR; // number of variables in set
            const uint_t _sizeIn, _sizeOut;
            const int buf_id;
            uint_t Nxghost, Nyghost; // may change depending on last chunk

            // Tmp storage for GPU input data
            cuda_vector_t GPUin_all;
            real_vector_t GPUin;

            // Tmp storage for GPU output data
            cuda_vector_t GPUout_all;
            real_vector_t GPUout;

            // compact ghosts
            cuda_vector_t xyghost_all;
            real_vector_t xghost_l, xghost_r, yghost_l, yghost_r;

            HostBuffer(const uint_t sizeIn, const uint_t sizeOut, const uint_t sizeXghost, const uint_t sizeYghost, const int b_ID=0) :
                _sizeIn(sizeIn), _sizeOut(sizeOut), buf_id(b_ID),
                Nxghost(sizeXghost), Nyghost(sizeYghost),
                GPUin_all(NVAR*sizeIn, 0.0), GPUin(NVAR, NULL),
                GPUout_all(NVAR*sizeOut, 0.0), GPUout(NVAR, NULL),
                xyghost_all(2*NVAR*sizeXghost + 2*NVAR*sizeYghost, 0.0),
                xghost_l(NVAR, NULL), xghost_r(NVAR, NULL),
                yghost_l(NVAR, NULL), yghost_r(NVAR, NULL)
            {
                for (uint_t i = 0; i < NVAR; ++i)
                {
                    GPUin[i]  = &GPUin_all[i * sizeIn];
                    GPUout[i] = &GPUout_all[i * sizeOut];
                }
                realign_ghost_pointer(sizeXghost, sizeYghost);
            }

            // --> CURRENTLY NOT USED ELSEWHERE, THEREFORE REQUIRES sizeZ % nslices == 0
            // this is dangerous!  Realign the ghost buffer pointers which
            // allows to copy a reduced buffer size to the GPU, e.g. if nslices_last !=
            // 0 for the last chunk in the queue. (Data in the buffers will be
            // coalesced at all times.  Ghosts are stored all in the same
            // contiguous array for better H2D/D2H bandwidth)
            void realign_ghost_pointer(const uint_t sizeXg, const uint_t sizeYg)
            {
                assert(sizeXg  <= Nxghost);
                assert(sizeYg  <= Nyghost);

                const uint_t allXghost = 2*NVAR*sizeXg;
                for (uint_t i = 0; i < NVAR; ++i)
                {
                    xghost_l[i] = &xyghost_all[(0*NVAR + i) * sizeXg];
                    xghost_r[i] = &xyghost_all[(1*NVAR + i) * sizeXg];
                    yghost_l[i] = &xyghost_all[(0*NVAR + i) * sizeYg + allXghost];
                    yghost_r[i] = &xyghost_all[(1*NVAR + i) * sizeYg + allXghost];
                }
            }
        };

        // using 2 buffers for additional CPU overlap
        HostBuffer BUFFER1, BUFFER2;
        HostBuffer *curr_buffer; // active buffer
        HostBuffer *prev_buffer;
        inline void _swap_buffer() // switch active buffer
        {
            curr_buffer = ((prev_buffer = curr_buffer) == &BUFFER1 ? &BUFFER2 : &BUFFER1);
        }


        ///////////////////////////////////////////////////////////////////////
        // PRIVATE HELPER
        ///////////////////////////////////////////////////////////////////////
        void _alloc_GPU();
        void _free_GPU();
        inline void _syncGPU() { GPU::syncGPU(); }
        inline void _syncStream(const int chunk_id) { GPU::syncStream(chunk_id); }
        void _reset();
        void _init_next_chunk();
        void _dump_chunk(const int complete = 0);

        inline void _copy_range(real_vector_t& dst, const uint_t dstOFFSET, const real_vector_t& src, const uint_t srcOFFSET, const uint_t Nelements)
        {
            for (int i = 0; i < GridMPI::NVAR; ++i)
                memcpy(dst[i] + dstOFFSET, src[i] + srcOFFSET, Nelements*sizeof(Real));
        }

        inline void _CONV_copy_range(real_vector_t& dst, const uint_t dstOFFSET, const real_vector_t& src, const uint_t srcOFFSET, const uint_t Nelements)
        {
            // primitive dest
            Real * const p_r = &(dst[0])[dstOFFSET];
            Real * const p_u = &(dst[1])[dstOFFSET];
            Real * const p_v = &(dst[2])[dstOFFSET];
            Real * const p_w = &(dst[3])[dstOFFSET];
            Real * const p_e = &(dst[4])[dstOFFSET];
            Real * const p_G = &(dst[5])[dstOFFSET];
            Real * const p_P = &(dst[6])[dstOFFSET];

            // conservative source
            const Real * const c_r = &(src[0])[srcOFFSET];
            const Real * const c_u = &(src[1])[srcOFFSET];
            const Real * const c_v = &(src[2])[srcOFFSET];
            const Real * const c_w = &(src[3])[srcOFFSET];
            const Real * const c_e = &(src[4])[srcOFFSET];
            const Real * const c_G = &(src[5])[srcOFFSET];
            const Real * const c_P = &(src[6])[srcOFFSET];

            memcpy(p_r, c_r, Nelements*sizeof(Real));
            memcpy(p_G, c_G, Nelements*sizeof(Real));
            memcpy(p_P, c_P, Nelements*sizeof(Real));

            for (int i = 0; i < Nelements; ++i)
            {
                const Real r = c_r[i];
                const Real u = c_u[i];
                const Real v = c_v[i];
                const Real w = c_w[i];
                const Real e = c_e[i];
                const Real G = c_G[i];
                const Real P = c_P[i];

                // convert
                p_u[i] = u/r;
                p_v[i] = v/r;
                p_w[i] = w/r;
                p_e[i] = (e - static_cast<Real>(0.5)*(u*u + v*v + w*w)/r - P) / G;
            }
        }

        inline void _copy_xyghosts() // alternatively, copy ALL x/yghosts at beginning
        {
            // copy from the halos into the ghost buffer of the current chunk
            _copy_range(curr_buffer->xghost_l, 0, halox.left,  3*sizeY*curr_iz, curr_buffer->Nxghost);
            _copy_range(curr_buffer->xghost_r, 0, halox.right, 3*sizeY*curr_iz, curr_buffer->Nxghost);
            _copy_range(curr_buffer->yghost_l, 0, haloy.left,  3*sizeX*curr_iz, curr_buffer->Nyghost);
            _copy_range(curr_buffer->yghost_r, 0, haloy.right, 3*sizeX*curr_iz, curr_buffer->Nyghost);
        }

        inline void _CONV_copy_xyghosts() // convert to primitive vars and copy xyghosts
        {
            _CONV_copy_range(curr_buffer->xghost_l, 0, halox.left,  3*sizeY*curr_iz, curr_buffer->Nxghost);
            _CONV_copy_range(curr_buffer->xghost_r, 0, halox.right, 3*sizeY*curr_iz, curr_buffer->Nxghost);
            _CONV_copy_range(curr_buffer->yghost_l, 0, haloy.left,  3*sizeX*curr_iz, curr_buffer->Nyghost);
            _CONV_copy_range(curr_buffer->yghost_r, 0, haloy.right, 3*sizeX*curr_iz, curr_buffer->Nyghost);
        }

        // execution helper
        void _process_chunk_sos(const real_vector_t& src);
        void _process_chunk_flow(const Real a, const Real b, const Real dtinvh, real_vector_t& src, real_vector_t& tmp);

        // info
        void _print_array(const Real * const in, const uint_t size);
        void _show_feature();
        void _start_info_current_chunk(const std::string title = "");
        inline void _end_info_current_chunk()
        {
            printf("}\n");
        }


    protected:

        enum {SKIN, FLESH} myFeature[6];

        GridMPI& grid;

        // TODO: This data should go into the host_buffer struct
        // CHUNK metrics
        uint_t curr_slices;
        uint_t curr_iz;
        uint_t curr_chunk_id;
        uint_t prev_slices;
        uint_t prev_iz;
        uint_t prev_chunk_id;

        // ghosts
        Halo halox, haloy, haloz;

        // boundary conditions (applied for myFeature == SKIN)
        virtual void _apply_bc(const double t = 0) {}


    public:

        GPUlabMPI(GridMPI& G, const uint_t nslices, const int verbosity=0, const bool isroot_=true);
        virtual ~GPUlabMPI() { _free_GPU(); }

        ///////////////////////////////////////////////////////////////////////
        // PUBLIC ACCESSORS
        ///////////////////////////////////////////////////////////////////////
        void load_ghosts(const double t = 0);
        /* inline void dump_chunk(const int complete = 0) { _dump_chunk(complete); } */
        double max_sos(float& sos);
        double process_all(const Real a, const Real b, const Real dtinvh);

        // info
        inline uint_t number_of_chunks() const { return nchunks; }
        inline uint_t chunk_slices() const { return curr_slices; }
        inline uint_t chunk_start_iz() const { return curr_iz; }
        inline uint_t chunk_id() const { return curr_chunk_id; }
        static inline Profiler& get_profiler() { return GPU::profiler; }
};

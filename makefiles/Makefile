SHELL := /bin/bash

CC ?= mpicxx
LD ?= mpicxx

NVCC ?= nvcc

include ../Makefile.config

mpi-inc ?=.
mpi-lib ?=.
fftw-inc ?=.
fftw-lib ?=.

ifneq "$(findstring silvio,$(shell hostname))" ""
	mpi-inc = /home/menahel/openmpi-1.5.5/include/
	mpi-lib = /home/menahel/openmpi-1.5.5/lib/
	ifeq "$(CC)" "icc"
		fftw-inc = /home/hbabak/fftw-2.1.5_icc/include/
		fftw-lib = /home/hbabak/fftw-2.1.5_icc/lib/
	else
                fftw-inc = /home/hbabak/fftw-2.1.5_gcc/include/
                fftw-lib = /home/hbabak/fftw-2.1.5_gcc/lib/
	endif
endif

ifneq "$(findstring brutus,$(shell hostname))" ""
	mpi-inc = ${MPI_ROOT}/include/
	mpi-lib = ${MPI_ROOT}/lib/
	fftw-inc = /cluster/work/infk/hbabak/apps/fftw-2.1.5/include/
	fftw-lib = /cluster/work/infk/hbabak/apps/fftw-2.1.5/lib/
endif

ifneq "$(findstring euler,$(shell hostname))" ""
        mpi-inc = /cluster/apps/openmpi/1.6.5/x86_64/gcc_4.8.2//include/
        mpi-lib = /cluster/apps/openmpi/1.6.5/x86_64/gcc_4.8.2//lib/
endif

ifneq "$(findstring rosa,$(shell hostname))" ""
	mpi-inc = $(CRAY_MPICH2_DIR)/include/
	mpi-lib = $(CRAY_MPICH2_DIR)/lib/
	fftw-inc = $(FFTW_INC)#/users/petrosk/fftw-2.1.5_gcc/include/
	fftw-lib = $(FFTW_DIR)#/users/petrosk/fftw-2.1.5_gcc/lib/
endif

ifneq "$(findstring daint,$(shell hostname))" ""
	mpi-inc = $(CRAY_MPICH2_DIR)/include/
	mpi-lib = $(CRAY_MPICH2_DIR)/lib/
	fftw-inc = $(FFTW_INC)#/users/petrosk/fftw-2.1.5_gcc/include/
	fftw-lib = $(FFTW_DIR)#/users/petrosk/fftw-2.1.5_gcc/lib/
endif

ifneq "$(findstring localhost.localdomain,$(shell hostname))" "" #Cyrus
	mpi-inc = /usr/include/openmpi-x86_64
	mpi-lib = /usr/lib64/openmpi/lib
	LIBS += -L/usr/local/cuda-6.0/lib64 -lcudart
endif

# CUFLAGS += -arch=sm_35 -Xptxas -v
# CUFLAGS += -arch=sm_35 -Xptxas -v -maxrregcount=64
# CUFLAGS += -arch=sm_35 -Xptxas -v -maxrregcount=50
CUFLAGS += -arch=sm_35 -Xptxas -v -maxrregcount=32

#CPPFLAGS += -D_ALIGNBYTES_=$(align) -D_BLOCKSIZE_=$(bs) -D_BLOCKSIZEX_=$(bs) -D_BLOCKSIZEY_=$(bs) -D_BLOCKSIZEZ_=$(bs) -DWENOEPS=$(weps) -DNTHREADS=$(nthreads) $(extra)
CPPFLAGS +=  -I$(mpi-inc)

ifneq "$(findstring rosa,$(shell hostname))" ""
	ifeq "$(CC)" "icc"
		LIBS += -L$(mpi-lib) -lmpich_intel -lmpichcxx_intel
	else
		LIBS += -L$(mpi-lib) -lmpich -lmpichcxx
	endif
else ifneq "$(findstring daint,$(shell hostname))" ""
        ifeq "$(CC)" "icc"
                LIBS += -L$(mpi-lib) -lmpich_intel -lmpichcxx_intel
        else
                LIBS += -L$(mpi-lib) -lmpich -lmpichcxx
        endif
else
	ifeq "$(findstring mpi,$(CC))" ""
		LIBS += -L$(mpi-lib) -lmpi -lmpi_cxx
	endif

#LIBS += -L$(mpi-lib) -lmpi -lmpi_cxx
endif

ifeq "$(fftw)"  "1"
#FFTW always in double precision
	CPPFLAGS += -I$(fftw-inc) -D_USE_FFTW_
	LIBS += -L$(fftw-lib) -lfftw3 -lfftw3_mpi #-ldfftw -ldfftw_mpi
endif

##################
VPATH := ../source/ ../source/WaveletCompression ../source/IO ../source/GPU ../source/Sim
.DEFAULT_GOAL := mpcf-cluster

# core
OBJECTS = main.o Types.o NodeBlock.o GPUlabMPI.o LSRK3_IntegratorMPI.o
# kernels
OBJECTS += MaxSpeedOfSound.o MaxSpeedOfSound_CUDA.o Convection_CUDA.o Update_CPP.o
# Simulations
OBJECTS += Sim_SteadyStateMPI.o Sim_SodMPI.o Sim_2DSBIMPI.o Sim_StaticIC.o Sim_SICCloudMPI.o
# CUDA
OBJECTS += cudaHostAllocator.o GPUhousehold.o GPUkernels.o Profiler.o

ifeq "$(qpxemu)" "1"
	OBJECTS += Update_QPX.o
endif

all: mpcf-cluster

mpcf-cluster: $(OBJECTS)
	$(CC) $(OPTFLAGS) $(extra) $^ -o $@ $(LIBS)

%.o: %.cpp
	$(CC) $(OPTFLAGS) $(CPPFLAGS) -c $^ -o $@

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $^ -o $@

GPUkernels: ../source/GPU/GPUkernels.cu
	$(NVCC) $(CUFLAGS) -c ../source/GPU/GPUkernels.cu -o GPUkernels.o

show:
	@echo "CC       = $(CC)"
	@echo "OBJECTS  = $(OBJECTS)"
	@echo "CPPFLAGS = $(CPPFLAGS)"
	@echo "OPTFLAGS = $(OPTFLAGS)"
	@echo "CUFLAGS  = $(CUFLAGS)"
	@echo "CUOPTFLAGS = $(CUOPTFLAGS)"
	@echo "LIBS     = $(LIBS)"
	@echo "EXTRA    = $(extra)"

clean:
	rm -f *.o mpcf-cluster *~

cleanall:
	make clean;


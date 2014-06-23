hdf ?= 1

ifneq "$(findstring brutus,$(shell hostname))" ""
	vtk-inc = /cluster/work/infk/cconti/VTK5.8_gcc/include/vtk-5.8/
	vtk-lib = /cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/
	#ifneq "$(findstring mpi,$(CC))" ""
	hdf-inc = $(HDF5_ROOT)/include/
	hdf-lib = $(HDF5_ROOT)/lib/
	#else
	#	hdf-inc = /cluster/work/infk/hbabak/apps/hdf5-1.8.8_gcc_serial/include/
	#	hdf-lib = /cluster/work/infk/hbabak/apps/hdf5-1.8.8_gcc_serial/lib/
	#endif
	numa-inc = /cluster/work/infk/hbabak/numactl-2.0.7/include/
	numa-lib = /cluster/work/infk/hbabak/numactl-2.0.7/
endif

ifeq "$(hdf)" "1"
	CPPFLAGS += -I$(hdf-inc) -D_USE_HDF_
	LIBS += -L$(hdf-lib) -lhdf5
endif

CPPFLAGS += -std=c++11 -fopenmp $(extra)

all: test.cpp NodeBlock.h NodeBlock.cpp WaveletCompressor.o
	mpic++ $(CPPFLAGS) test.cpp NodeBlock.cpp WaveletCompressor.o -o test $(LIBS) -lz

WaveletCompressor.o: WaveletCompressor.h WaveletCompressor.cpp
	g++ $(CPPFLAGS) $(extra)  -c WaveletCompressor.cpp -o WaveletCompressor.o

clean:
	rm -f lsf.* ghosts_rank_* rank_* *~ *.o test

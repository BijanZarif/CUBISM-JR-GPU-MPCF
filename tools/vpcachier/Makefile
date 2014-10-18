bs ?= 16
vs ?= 128

MYFLAGS =-D_USE_ZLIB_ -D_ALIGNBYTES_=16 -I../lz4 -mtune=native -march=native -ftree-vectorize -fstrict-aliasing -D_BLOCKSIZE_=$(bs) -D_SUBBLOCKSIZE_=$(bs) -D_VOXELS_=$(vs) -D_FLOAT_PRECISION_ -D_SP_COMP_ -g -fopenmp -I../reader/ -I../../source/IO -I../../source/WaveletCompression -Ofast -D_QPXEMU_

ifneq "$(findstring daint,$(shell hostname))" ""
        MYFLAGS +=-I$(CRAY_MPICH2_DIR)/include/ -L$(CRAY_MPICH2_DIR)/lib/ -lmpich -lmpichcxx
endif

vpcachier: main.o WaveletCompressor.o
	$(CC) WaveletCompressor.o  main.o -o vpcachier  $(MYFLAGS) -lz -lstdc++

main.o: main.cpp
	$(CC) $(MYFLAGS) -c main.cpp -o main.o

WaveletCompressor.o: ../../source/WaveletCompression/WaveletCompressor.cpp
	$(CC) $(MYFLAGS) -c ../../source/WaveletCompression/WaveletCompressor.cpp -S
	$(CC) $(MYFLAGS) -c ../../source/WaveletCompression/WaveletCompressor.cpp -o WaveletCompressor.o

clean:
	rm vpcachier *.o

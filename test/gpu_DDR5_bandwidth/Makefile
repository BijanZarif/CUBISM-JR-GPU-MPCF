#  File        : Makefile
#  Maintainer  : Fabian Wermelinger <fabianw@student.ethz.ch>
#  Created     : Wed 30 Jul 2014 09:32:16 AM CEST
#  Modified    : Never
#  Description :
CC = nvcc
all: bw_bench.cu
	$(CC) -arch=sm_35 -O3 bw_bench.cu -o bw_bench

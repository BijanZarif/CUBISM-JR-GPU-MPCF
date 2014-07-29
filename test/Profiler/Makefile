#  File        : Makefile
#  Maintainer  : Fabian Wermelinger <fabianw@student.ethz.ch>
#  Created     : Tue 29 Jul 2014 02:42:50 PM CEST
#  Modified    : Never
#  Description :


Test_Profiler: Test_Profiler.cu ../../source/Profiler.cu
	nvcc -arch=sm_35 $(extra) -I../../source Test_Profiler.cu ../../source/Profiler.cu -o Test_Profiler

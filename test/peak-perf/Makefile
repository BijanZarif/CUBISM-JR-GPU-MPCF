SRC = peak-perf-pure.cu

CC = nvcc

all: $(SRC)
	# $(CC) $(extra) -arch=sm_35 -Xptxas -v,-dlcm=cg $(SRC) -o peak-perf
	$(CC) $(extra) -O3 -ftz=true -arch=sm_35 -Xptxas -v $(SRC) -o peak-perf

sass: all
	cuobjdump --dump-sass peak-perf > sass-code

ptx: all
	cuobjdump --dump-ptx peak-perf > ptx-code

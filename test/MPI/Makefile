all: test.cpp NodeBlock.h NodeBlock.cpp
	mpic++ -fopenmp $(extra) test.cpp NodeBlock.cpp -o test

NodeBlock.o: NodeBlock.h NodeBlock.cpp
	g++ -c NodeBlock.cpp -o NodeBlock.o

clean:
	rm -f lsf.* ghosts_rank_* rank_* *~ test

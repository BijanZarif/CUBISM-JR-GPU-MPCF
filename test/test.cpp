/* *
 * test.cpp
 *
 * Created by Fabian Wermelinger on 6/18/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */

#include <mpi.h>
#include <stdio.h>


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const bool isroot = rank == 0;

    if (isroot) printf("Comm size is %d\n", size);
    printf("Hi from rank %d\n", rank);


    MPI_Finalize();

    return 0;
}

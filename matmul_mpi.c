#include <mpi.h>
#include <stdio.h>

/*
* This main function calculates a parallel matrix multiplication.
*/
int main(int argc, char **argv) {

    // Initializing the environment
    MPI_Init(&argc, &argv);

    /*
     * rank - the id of the process
     * size - number of processes
     */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Closing the MPI
    MPI_Finalize();
    return 0;
}
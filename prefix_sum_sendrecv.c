#include <mpi.h>
#include <stdio.h>

/*
* This main function calculates a Parallel prefix sum.
* Each process has a rank between 0 and P - 1 where P is the number of processes
* The processes will calculate the prefix sum of the array [0,...,P - 1] in parallel using MPI
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
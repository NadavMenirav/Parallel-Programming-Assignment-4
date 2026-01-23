#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

/*
* This main function calculates a parallel matrix multiplication.
*/
int main(int argc, char **argv) {

    // 4 arguments + program name
    if (argc != 5) {
        return 1;
    }

    // Initializing the environment
    MPI_Init(&argc, &argv);

    /*
     * rank - the id of the process
     * size - number of processes
     */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Extracting the command line arguments
    const char* N_char = argv[1];
    const char* seedA_char = argv[2];
    const char* seedB_char = argv[3];
    const char* maxValue_char = argv[4];

    // Casting to integers
    const int N = atoi(N_char);
    const int seedA = atoi(seedA_char);
    const int seedB = atoi(seedB_char);
    const int maxValue = atoi(maxValue_char);

    // Allocating the memory for the matrices
    IMatrix A = imatrix_alloc(N);
    IMatrix B = imatrix_alloc(N);

    // Filling the matrices with random values - only for rank = 0
    if (rank == 0) {
        imatrix_fill_random(&A, seedA, maxValue);
        imatrix_fill_random(&B, seedB, maxValue);
    }

    /*
     * Broadcasting B matrix to everyone.
     * Since Imatrix is not a valid MPI type - we only pass the values in 'data'
     */
    MPI_Bcast(B.data, N * N, MPI_INT, 0, MPI_COMM_WORLD);


    // Closing the MPI
    MPI_Finalize();
    return 0;
}
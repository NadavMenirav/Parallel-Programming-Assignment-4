#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

/*
 * This main function calculates a parallel matrix multiplication using 1D Row-Block Distribution
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


    // Parameters for the scatterv function that is used to send the different rows of A

    // Allocating the sendcounts and displs arrays
    int* sendCounts = calloc(size, sizeof(int));
    int* displs = calloc(size, sizeof(int));

    /*
     * Process 0 calculates the number of elements to send to each process using the provided formula.
     * Also calculates the first row each process gets
     */
    if (rank == 0) {
        for (int i = 0; i < size; i++) {

            // Using the provided formula to calculate the first row for each process
            const int first_row = i * N / size;

            // The first integer in the row is at index row_number * N
            displs[i] = N * first_row;

            // Calculating the number of rows each process gets
            const int last_row = (i + 1) * N / size - 1;
            const int row_count = last_row - first_row + 1;

            // The number of elements each process gets is row_count times the number of elements in each row (N)
            sendCounts[i] = row_count * N;
        }
    }

    // Closing the MPI
    MPI_Finalize();

    free(sendCounts);
    free(displs);
    return 0;
}
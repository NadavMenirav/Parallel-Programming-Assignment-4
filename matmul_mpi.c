#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

// Calculates the number of rows each process receives
int get_rows_for_rank(int r, int N, int P);

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
            sendCounts[i] = get_rows_for_rank(i, N, size) * N;
        }
    }

    // Now calculating locally for each process the number of integers he receives so it can allocate its memory
    const int number_of_integers = get_rows_for_rank(rank, N, size) * N;



    // Closing the MPI
    MPI_Finalize();

    free(sendCounts);
    free(displs);
    return 0;
}

// r - rank, N - dimension of the space of the matrix, P - number of processes
int get_rows_for_rank(const int r, const int N, const int P) {
    const int first_row = r * N / P;
    const int last_row = (r + 1) * N / P - 1;
    const int row_count = last_row - first_row + 1;
    return row_count;
}

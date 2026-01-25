#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sum = rank;
    int received_val = 0;

    for (int step = 1; step < size; step *= 2) {
        const int send_to = (rank + step < size) ? (rank + step) : MPI_PROC_NULL;
        const int recv_from = (rank - step >= 0) ? (rank - step) : MPI_PROC_NULL;

        // Sending and receiving in one atomic command
        MPI_Sendrecv(&sum, 1, MPI_INT, send_to, 0,
                     &received_val, 1, MPI_INT, recv_from, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (recv_from != MPI_PROC_NULL) {
            sum += received_val;
        }
    }

    printf("rank=%d x=%d prefix=%d\n", rank, rank, sum);

    MPI_Finalize();
    return 0;
}
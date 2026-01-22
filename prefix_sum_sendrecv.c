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

    // In Prefix Sum, we start off with an array identical to the original array
    int sum = rank;

    for (int step = 1; step < size; step *= 2) {
        // The process that will receive our current sum
        const int send_to = rank + step;
        const int recv_from = rank - step;

        // We only send the message if the target is a valid Process
        if (send_to < size) {
            MPI_Send(&sum, 1, MPI_INT, send_to, step, MPI_COMM_WORLD);
        }

        // We will wait for a message if the process sending us is actually a valid process
        if (recv_from >= 0 && recv_from < size) {
            int message_received = 0;
            MPI_Recv(&message_received, 1, MPI_INT, recv_from, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += message_received;
        }
    }

    printf("rank=%d x=%d prefix=%d\n", rank, rank, sum);

    // Closing the MPI
    MPI_Finalize();
    return 0;
}
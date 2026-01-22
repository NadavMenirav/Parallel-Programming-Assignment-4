cmake_minimum_required(VERSION 4.1)
project(assignment_4 C)

set(CMAKE_C_STANDARD 11)

include_directories(/opt/homebrew/include)
link_directories(/opt/homebrew/lib)

find_package(MPI REQUIRED)

add_executable(mpi_exec prefix_sum_sendrecv.c)

target_link_libraries(mpi_exec MPI::MPI_C)
# 🚀 MPI Parallel Programming

**Bar-Ilan CS 89-3312: Parallel Systems Programming**  
*Assignment 4 - Part 1: MPI*

---

## 🎯 Overview

This project implements two fundamental parallel algorithms using **MPI (Message Passing Interface)**:

1. **Parallel Prefix Sum** - Computing cumulative sums across distributed processes
2. **Parallel Matrix Multiplication** - Distributed matrix computation using 1D row-block distribution

Both implementations demonstrate distributed memory parallelism and inter-process communication patterns.

---

## 📊 Task 1: Parallel Prefix Sum

### Objective
Compute an inclusive prefix sum across MPI ranks using point-to-point communication.

### Problem Definition
Given `P` processes with ranks `r ∈ {0, ..., P-1}`:
- Each process starts with: `xr = r`
- Compute on every rank: `yr = Σ(k=0 to r) xk`

### Algorithm
The implementation uses a **logarithmic step-doubling algorithm**:

```
Step 1: Each process sends to rank+1, receives from rank-1
Step 2: Each process sends to rank+2, receives from rank-2
Step 4: Each process sends to rank+4, receives from rank-4
...
```

**Example with 8 processes:**
```
Initial:  [0, 1, 2, 3, 4, 5, 6, 7]

After step 1: [0, 1, 3, 5, 7, 9, 11, 13]
After step 2: [0, 1, 3, 6, 10, 14, 18, 22]
After step 4: [0, 1, 3, 6, 10, 15, 21, 28]

Final:    [0, 1, 3, 6, 10, 15, 21, 28]
```

### Running
```bash
mpirun -np <P> ./mpi_exec
```

**Example:**
```bash
mpirun -np 8 ./mpi_exec
```

### Output Format
Each rank prints exactly one line:
```
rank=<r> x=<x_r> prefix=<y_r>
```

**Example output (4 processes):**
```
rank=0 x=0 prefix=0
rank=1 x=1 prefix=1
rank=2 x=2 prefix=3
rank=3 x=3 prefix=6
```

### Requirements
- ✅ Works for any P ≥ 1
- ✅ Uses only `MPI_Send` and `MPI_Recv` (no `MPI_Scan`)
- ✅ Values computed through communication, not local loops
- ✅ Time complexity: O(log P)

---

## 🔢 Task 2: Parallel Matrix Multiply

### Objective
Implement distributed matrix multiplication `C = A × B` using MPI with **1D row-block distribution**.

### Problem Definition
- Input: Two N×N matrices A and B
- Output: Matrix C = A × B
- Distribution: Rows of A (and C) are partitioned across P processes

### Algorithm

#### 1. **Data Distribution**
Rows are partitioned using the formula:
```
Process r handles rows: [r*N/P, (r+1)*N/P - 1]
```

**Example (N=10, P=3):**
```
Process 0: rows 0-2   (3 rows)
Process 1: rows 3-5   (3 rows)
Process 2: rows 6-9   (4 rows)
```

#### 2. **Execution Flow**

```
┌──────────────────────────────────────────┐
│ Rank 0: Generate A and B                 │
│         (using seedA and seedB)          │
└──────────────────┬───────────────────────┘
                   │
         ┌─────────┴──────────┐
         │   MPI_Scatterv     │  Distribute A rows
         │   (A → all ranks)  │
         └─────────┬──────────┘
                   │
         ┌─────────┴──────────┐
         │    MPI_Bcast       │  Send full B to all
         │   (B → all ranks)  │
         └─────────┬──────────┘
                   │
         ┌─────────┴──────────┐
         │ Each rank computes │  Local multiplication
         │   its C_local      │  (triple-loop algorithm)
         └─────────┬──────────┘
                   │
         ┌─────────┴──────────┐
         │   MPI_Gatherv      │  Collect results
         │   (C_local → C)    │
         └─────────┬──────────┘
                   │
         ┌─────────┴──────────┐
         │ Rank 0: Compute    │
         │  checksum(C)       │
         └────────────────────┘
```

### Running
```bash
mpirun -np <P> ./matmul <N> <seedA> <seedB> <maxValue>
```

**Parameters:**
- `N` - Matrix dimension (N × N)
- `seedA` - Random seed for matrix A
- `seedB` - Random seed for matrix B
- `maxValue` - Maximum value in matrices (values in range [0, maxValue])

**Example:**
```bash
# 10×10 matrices, 4 processes
mpirun -np 4 ./matmul 10 123 456 100

# 100×100 matrices, 8 processes
mpirun -np 8 ./matmul 100 42 1337 50
```

### Output
Rank 0 prints the checksum:
```
checksum(C)=<value>
```

**Example:**
```
checksum(C)=1234567
```

### Matrix Library
The implementation uses the provided `matrix.h` and `matrix.c` library:

**Key Functions:**
```c
IMatrix imatrix_alloc(int N);                          // Allocate N×N matrix
void imatrix_fill_random(IMatrix *M, uint64_t seed, 
                         int max_value);               // Fill with random values
long long imatrix_checksum(const IMatrix *M);          // Compute checksum
void imatrix_free(IMatrix *M);                         // Free memory
```

### Requirements
- ✅ Works for any N ≥ 1 and P ≥ 1
- ✅ Handles cases where N is not divisible by P
- ✅ Rank 0 holds complete C matrix at the end
- ✅ Deterministic results (same seeds → same output)
- ✅ Uses `MPI_Scatterv` and `MPI_Gatherv` for irregular distributions

---

## 🛠️ Building and Running

### Using CMake

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make

# This creates two executables:
#   - mpi_exec  (prefix sum)
#   - matmul    (matrix multiply)
```

### Running the Programs

**Prefix Sum:**
```bash
mpirun -np 8 ./mpi_exec
```

**Matrix Multiplication:**
```bash
# Small test (4×4 matrix, 2 processes)
mpirun -np 2 ./matmul 4 100 200 10

# Medium test (50×50 matrix, 4 processes)
mpirun -np 4 ./matmul 50 12345 67890 100

# Large test (1000×1000 matrix, 16 processes)
mpirun -np 16 ./matmul 1000 42 1337 50
```

### Testing Edge Cases

```bash
# Single process
mpirun -np 1 ./mpi_exec
mpirun -np 1 ./matmul 10 1 2 100

# Non-divisible N/P
mpirun -np 3 ./matmul 10 5 10 50  # 10 rows, 3 processes
mpirun -np 7 ./matmul 20 7 14 25  # 20 rows, 7 processes
```

---

## 📁 Project Structure

```
mpi_programs/
├── CMakeLists.txt              # Build configuration
│
├── prefix_sum_sendrecv.c       # Task 1: Parallel prefix sum
│
├── matmul_mpi.c                # Task 2: Matrix multiplication
├── matrix.h                    # Matrix library header
└── matrix.c                    # Matrix library implementation
```

---

## 🔍 Implementation Details

### Prefix Sum (`prefix_sum_sendrecv.c`)

**Key Algorithm Points:**
- Uses logarithmic step doubling: step = 1, 2, 4, 8, ...
- Each process sends to `rank + step` and receives from `rank - step`
- Guards against invalid ranks with boundary checks
- Accumulates received values into local sum

**Communication Pattern:**
```c
for (int step = 1; step < size; step *= 2) {
    send_to = rank + step;
    recv_from = rank - step;
    
    if (send_to < size)
        MPI_Send(&sum, 1, MPI_INT, send_to, step, MPI_COMM_WORLD);
    
    if (recv_from >= 0)
        MPI_Recv(&message, 1, MPI_INT, recv_from, step, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sum += message;
}
```

### Matrix Multiply (`matmul_mpi.c`)

**Key Implementation Points:**

1. **Row Distribution Calculation:**
```c
int get_rows_for_rank(int r, int N, int P) {
    int first_row = r * N / P;
    int last_row = (r + 1) * N / P - 1;
    return last_row - first_row + 1;
}
```

2. **Scatterv Setup (Rank 0):**
```c
for (int i = 0; i < size; i++) {
    displs[i] = (i * N / size) * N;           // Starting position
    sendCounts[i] = get_rows_for_rank(i, N, size) * N;  // Elements to send
}
```

3. **Local Multiplication:**
```c
// Standard triple-loop: C[i,j] = Σ A[i,k] * B[k,j]
for (int i = 0; i < rowA; i++)
    for (int j = 0; j < colB; j++)
        for (int k = 0; k < colA; k++)
            C[i * colB + j] += A[i * colA + k] * B[k * colB + j];
```

4. **Memory Management:**
   - Rank 0: Allocates full A, B, and C matrices
   - Other ranks: Allocate only their row blocks and full B
   - All ranks: Free allocated memory before finalization

---

## 📚 MPI Functions Used

| Function | Purpose |
|----------|---------|
| `MPI_Init` | Initialize MPI environment |
| `MPI_Finalize` | Clean up MPI environment |
| `MPI_Comm_rank` | Get process rank |
| `MPI_Comm_size` | Get total number of processes |
| `MPI_Send` | Point-to-point send |
| `MPI_Recv` | Point-to-point receive |
| `MPI_Bcast` | Broadcast data to all processes |
| `MPI_Scatterv` | Scatter variable-sized data |
| `MPI_Gatherv` | Gather variable-sized data |

---

## ✅ Correctness Verification

### Prefix Sum
Verify with formula: `prefix[r] = r * (r + 1) / 2`

**Example (P=5):**
```
rank=0: prefix=0  ✓ (0*1/2 = 0)
rank=1: prefix=1  ✓ (1*2/2 = 1)
rank=2: prefix=3  ✓ (2*3/2 = 3)
rank=3: prefix=6  ✓ (3*4/2 = 6)
rank=4: prefix=10 ✓ (4*5/2 = 10)
```

### Matrix Multiply
- Verify checksum consistency with same seeds
- Test with small matrices (can verify manually)
- Compare parallel result with sequential computation

---

**Good luck! 🚀**

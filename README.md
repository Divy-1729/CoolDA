
# Understanding CUDA Threads, Blocks, and Grids Through Vector Addition and Matrix Multiplication

I finally got to understanding how CUDA parallelizes computation using **threads, blocks, and grids**, based on two examples:

1. Vector addition
2. Matrix multiplication

The goal is to understand how CUDA assigns pieces of a computation to individual GPU threads.

---

# 1. CUDA Execution Hierarchy

CUDA organizes work in three levels:

Grid  
└── Blocks  
  └── Threads  

- **Thread**: the smallest execution unit.
- **Block**: a group of threads that execute together.
- **Grid**: a collection of blocks that cover the full computation.

All threads run the **same kernel code**, but operate on **different indices of the data**.

---

# 2. Vector Addition Example

Suppose we want to compute:

```
C[i] = A[i] + B[i]
```

On a CPU we might write:

```

for (int i = 0; i < n; i++) {
C[i] = A[i] + B[i];
}

````
This runs **sequentially**, meaning the CPU computes:


i = 0 → compute
i = 1 → compute
i = 2 → compute
...


On a GPU we instead **assign one thread per index**.

So instead of looping over `i`, each thread computes its own value of `i`.

Then, all these indices add simultaneously.

| Thread | threadIdx.x | Computes              |
| ------ | ----------- | --------------------- |
| T0     | 0           | c[0] = a[0] + b[0]    |
| T1     | 1           | c[1] = a[1] + b[1]    |
| T2     | 2           | c[2] = a[2] + b[2]    |
| ...    | ...         | ...                   |
| T19    | 19          | c[19] = a[19] + b[19] |

This makes GPUs really good at parallel operation.

---

## Insight from Vector Addition

Each thread computes **one element of the output vector**.

The global index of a thread is computed as:

```
index = blockIdx.x * blockDim.x + threadIdx.x
```

This formula converts:

* the thread’s position **inside the block**
* plus the block’s position **inside the grid**

into a **global index in the data**.

Example:

```
blockDim.x = 8
blockIdx.x = 2
threadIdx.x = 3

index = 2*8 + 3 = 19
```

So that thread computes:

```
C[19] = A[19] + B[19]
```

---

# 3. Matrix Multiplication

Now consider matrix multiplication:

```
C = A × B
```

Where:

```
A is m × n
B is n × k
C is m × k
```

Each element of C is defined as:

```
C[i,j] = Σ A[i,t] * B[t,j]
```

This is the **dot product of row i of A with column j of B**.

---

# 4. CUDA Matrix Multiplication Kernel

```cpp
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                              int m, int n, int k) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < k) {
    float sum = 0.0f;

    for (int i = 0; i < n; ++i) {
      sum += A[row * n + i] * B[i * k + col];
    }

    C[row * k + col] = sum;
  }
}
```

---

# 5. Key CUDA Indexing Insight

The most important lines are:

```
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

These determine **which element of the output matrix C this thread computes**.

Each thread computes exactly **one (row, col) element of C**.

---

# 6. Numerical Example

Suppose we use:

```
blockDim = (2,2)
gridDim  = (2,2)
```

This means:

* each block contains **2×2 threads**
* the grid contains **4 blocks**

Together they cover a **4×4 output matrix**.

Output matrix layout:

```
C (4×4)

(0,0) (0,1) | (0,2) (0,3)
(1,0) (1,1) | (1,2) (1,3)
------------+------------
(2,0) (2,1) | (2,2) (2,3)
(3,0) (3,1) | (3,2) (3,3)
```

Each block computes a **tile of this matrix**.

---

# 7. Insight: Computing C(0,0)

Consider the thread:

```
blockIdx = (0,0)
threadIdx = (0,0)
```

Compute row:

```
row = blockIdx.y * blockDim.y + threadIdx.y
row = 0*2 + 0
row = 0
```

Compute column:

```
col = blockIdx.x * blockDim.x + threadIdx.x
col = 0*2 + 0
col = 0
```

So this thread computes:

```
C(0,0)
```

To compute C(0,0), the thread multiplies:

```
row 0 of A
with
column 0 of B
```

Which means:

```
C[0,0] =
A[0,0]*B[0,0] +
A[0,1]*B[1,0] +
A[0,2]*B[2,0] + ...
```

---

# 8. Insight: Computing C(0,1)

Now consider the thread:

```
blockIdx = (0,0)
threadIdx = (0,1)
```

Compute row:

```
row = 0*2 + 0
row = 0
```

Compute column:

```
col = 0*2 + 1
col = 1
```

So this thread computes:

```
C(0,1)
```

To compute C(0,1), the thread multiplies:

```
row 0 of A
with
column 1 of B
```

Which gives:

```
C[0,1] =
A[0,0]*B[0,1] +
A[0,1]*B[1,1] +
A[0,2]*B[2,1] + ...
```

---

# 9. Final Insight

The key insight is that **each CUDA thread computes one element of the output matrix**.

More precisely:

```
thread → one (row,col) of C
block → a small tile of C
grid → the entire matrix
```

And each thread performs the computation:

```
C[row,col] = dot( A[row,:], B[:,col] )
```

This allows thousands of GPU threads to compute many output elements **simultaneously**, making matrix multiplication highly parallelizable.


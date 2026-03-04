#include <iostream>
#include <cuda_runtime.h>

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

int main() {
  int m = 256;
  int n = 128;
  int k = 24;

  // Allocating memory placeholders
  float* h_A, *h_B, *h_C;
  h_A = (float*)malloc(m * n * sizeof(float));
  h_B = (float*)malloc(n * k * sizeof(float));
  h_C = (float*)malloc(m * k * sizeof(float));

  // We add random data here
  for (int i = 0; i < m * n; ++i) {
    h_A[i] = rand() / (float)RAND_MAX; 
  }
  for (int i = 0; i < n * k; ++i) {
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate device memory GPU
  float* d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, m * n * sizeof(float));
  cudaMalloc((void**)&d_B, n * k * sizeof(float));
  cudaMalloc((void**)&d_C, m * k * sizeof(float));

  // Copy data from host to device (CPU to GPU)
  cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, n * k * sizeof(float), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 blockDim(4, 4); 
  dim3 gridDim((k + blockDim.x - 1) / blockDim.x, 
               (m + blockDim.y - 1) / blockDim.y);

  // Launch the kernel parallelize in GridxBlock
  matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
  
  // Copy results from GPU back to CPU in (h_C)
  cudaMemcpy(h_C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Result matrix 10 elements: ";
  for (int i = 0; i < 10; ++i) {
    std::cout << h_C[i] << " ";
  }
  std::cout << std::endl;

  // Cleanup
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}

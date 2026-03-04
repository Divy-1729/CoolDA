#include <iostream>
#include <cuda_runtime.h>

// define our CUDA kernel
__global__ void vectorAdd(int *a, int *b, int *c, int n){
    int i = threadIdx.x;
    while (i < n){
        c[i] = a[i] + b[i];
        i += 1;
    }
}

int main() {
    const int n = 20;
    // init arrays
    int a[n], b[n], c[n];

    int *d_a, *d_b, *d_c;

   // allocate memory
   cudaMalloc((void**) &d_a, n*sizeof(int));
   cudaMalloc((void**) &d_b, n*sizeof(int));
   cudaMalloc((void**) &d_c, n*sizeof(int));

    for(int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i*2;
    }
    
    //populate from CPU to GPU

    cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);

    vectorAdd<<<1,n>>>(d_a, d_b, d_c, n);

    // Copy c from GPU to CPU

    cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);

    // free memory in pointers

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}

#define ARRAY_MIN 1024
#define ARRAY_MAX 1024
#define EPSILON 0.0001
#define BLOCK_SIZE 1024

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void kogge_stone_scan_multi_block_kernel(uint32_t *blockCount, uint32_t* blockFlags, float* blockSums, float* X, float* Y, uint32_t size){
    __shared__ float XY_0[BLOCK_SIZE];
    __shared__ float XY_1[BLOCK_SIZE];
    __shared__ float precedingBlockSum;
    __shared__ uint32_t blockIndex;
    if (threadIdx.x == 0) {
        precedingBlockSum = 0;
        blockIndex = atomicAdd(blockCount, 1);
    }
    __syncthreads();

    // dynamic block index to ensure earlier blocks are processed before later ones
    uint32_t i = blockIndex * BLOCK_SIZE + threadIdx.x;

    if (i < size) {
        XY_0[threadIdx.x] = X[i];
    } else {
        XY_0[threadIdx.x] = 0;
    }

    uint32_t k = 0;
    for (uint32_t stride = 1; stride < BLOCK_SIZE; stride *= 2){
        ++k;
        __syncthreads();
        if (threadIdx.x >= stride) {
            if (k % 2 == 1)
                XY_1[threadIdx.x] = XY_0[threadIdx.x] + XY_0[threadIdx.x - stride];
            else
                XY_0[threadIdx.x] = XY_1[threadIdx.x] + XY_1[threadIdx.x - stride];
        }
        else {
            if (k % 2 == 1)
            XY_1[threadIdx.x] = XY_0[threadIdx.x];
            else
                XY_0[threadIdx.x] = XY_1[threadIdx.x];
        }
    }

    // Wait for global flag to be set
    if (blockIndex > 0) {
        if (threadIdx.x == 0) {
            while(atomicAdd(blockFlags+blockIndex-1, 0) == 0)
                ;
            precedingBlockSum = blockSums[blockIndex - 1];
        }
        __syncthreads();
    }

    // Add value from block i - 1 to all elements and assign to output (will be zero if at blockIdx 0)
    if (i < size) {
        if (k % 2 == 1) 
            Y[i] = XY_1[threadIdx.x] + precedingBlockSum;
        else
            Y[i] = XY_0[threadIdx.x] + precedingBlockSum;
    }

    // Assign last thread's value to blockSums and turn blockFlags true
    if (threadIdx.x == BLOCK_SIZE - 1){
        if (k % 2 == 1) {
            blockSums[blockIndex] = XY_1[threadIdx.x] + precedingBlockSum;
        }
        else {
            blockSums[blockIndex] = XY_0[threadIdx.x] + precedingBlockSum;
        }
        __threadfence();
        blockFlags[blockIndex] = true;
    }
}

void inclusiveScanCPU(float* A, float* B, uint32_t size) {
    B[0] = A[0];
    for (size_t i = 1; i < size; ++i) {
        B[i] = B[i-1] + A[i];
    }
}

void fillIncreasing(float* array, uint32_t size) {
    for (size_t i = 0; i < size; ++i) {
        array[i] = i+1;
    }
}


bool isSameFloat(float lhs, float rhs) { return std::abs(lhs - rhs) <= EPSILON; }

int main(void) {
    uint32_t arraySize = (rand() % ARRAY_MAX) + ARRAY_MIN;
    uint32_t grid_size = int((arraySize - 1) / BLOCK_SIZE) + 1;


    // Allocate host memory
    float A_h[arraySize];
    float B_h[arraySize];
    float B_ref[arraySize];
    // Fill host source array
    fillIncreasing(A_h, arraySize);

    // Allocate device memory for source and result
    float* A_d;
    float* B_d;
    cudaMalloc(&A_d, arraySize*4);
    cudaMalloc(&B_d, arraySize*4);
    // Copy host source array to device source array
    cudaMemcpy(A_d, A_h, arraySize*4, cudaMemcpyHostToDevice);

    // Allocae device memory for blockFlags and blockSums
    uint32_t* blockCount_d;
    uint32_t* blockFlags_d;
    float*    blockSums_d;
    cudaMalloc(&blockCount_d, 4);
    cudaMalloc(&blockFlags_d, grid_size*4);
    cudaMalloc(&blockSums_d, grid_size*4);
    // Set all to zero
    cudaMemset(blockCount_d, 0, 4);
    cudaMemset(blockFlags_d, 0, grid_size*4);
    cudaMemset(blockSums_d, 0, grid_size*4);

    // Get CPU results for later comparison
    inclusiveScanCPU(A_h, B_ref, arraySize);

    // Get GPU results
    kogge_stone_scan_multi_block_kernel<<<grid_size, BLOCK_SIZE>>>(blockCount_d, blockFlags_d, blockSums_d, A_d, B_d, arraySize);
    cudaMemcpy(B_h, B_d, arraySize*4, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(blockCount_d);
    cudaFree(blockFlags_d);
    cudaFree(blockSums_d);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < 10; ++i) {
        std::cout << "B_ref[" << i << "]: " << B_ref[i] << "\tB_h[" << i << "]: " << B_h[i] << std::endl;
    }

    // Compare GPU and CPU to look for errors
    for (size_t i = 0; i < arraySize; ++i) {
        if (!isSameFloat(B_h[i], B_ref[i])) {
            std::cout << "CPU and GPU results are not equal!" << B_h[i] << std::endl;
            std::cout << "B_ref[" << i << "]: " << B_ref[i] << "\tB_h[" << i << "]: " << B_h[i] << std::endl;
            return 0;
        }
    }

    std::cout << "Scan working as expected!" << std::endl;
}
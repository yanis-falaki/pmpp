#define ARRAY_MIN 32
#define ARRAY_MAX 50000
#define BLOCK_SIZE 512

#include <iostream>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

__global__ void scan_block_kernel(float* XY, uint32_t size, float* auxiliaryArray = nullptr) {
    // Normal kogge stone double buffering
    __shared__ float XY_0[BLOCK_SIZE];
    __shared__ float XY_1[BLOCK_SIZE];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size) {
        XY_0[threadIdx.x] = XY[i];
    } else {
        XY_0[threadIdx.x] = 0.0f;
    }

    float* out = XY_0;
    float* in = XY_1;
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();      
        if (threadIdx.x >= stride)
            in[threadIdx.x] = out[threadIdx.x] + out[threadIdx.x - stride];
        else
            in[threadIdx.x] = out[threadIdx.x];

        float* tmp = in;
        in = out;
        out = tmp;
    }

    if (i < size)
        XY[i] = out[threadIdx.x];

    if (auxiliaryArray && threadIdx.x == BLOCK_SIZE - 1)
        auxiliaryArray[blockIdx.x] = out[threadIdx.x];
}

__global__ void add_scan_block_offsets_kernel(float* X, uint32_t size, float* auxiliaryArray) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && blockIdx.x > 0)
        X[i] += auxiliaryArray[blockIdx.x-1];
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

bool isSameFloat(float lhs, float rhs) { 
    float maxVal = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= maxVal * 1e-3f; // relative epsilon
}

int main(void) {
    //uint32_t arraySize = (rand() % (ARRAY_MAX-ARRAY_MIN)) + ARRAY_MIN;
    uint32_t arraySize = 50000;
    uint32_t grid_size = ceil(float(arraySize) / float(BLOCK_SIZE));


    // Allocate host memory
    float A_h[arraySize];
    float B_h[arraySize];
    float B_ref[arraySize];
    // Fill host source array
    fillIncreasing(A_h, arraySize);

    // Allocate device memory for source and result
    float* XY_d;
    cudaMalloc(&XY_d, arraySize*4);
    // Copy host source array to device source array
    cudaMemcpy(XY_d, A_h, arraySize*4, cudaMemcpyHostToDevice);

    // Get CPU results for later comparison
    inclusiveScanCPU(A_h, B_ref, arraySize);

    // START GPU implementation

    uint32_t numIterations = ceil(log(arraySize)/log(BLOCK_SIZE));
    float* arrayPointers[numIterations];
    arrayPointers[0] = XY_d;
    uint32_t arraySizes[numIterations];
    arraySizes[0] = arraySize;

    // Downsample block
    for (int i = 0; i < numIterations; ++i) {
        // If on the last iteration then grid_size < blockSize and we don;t need an auxiliary array.
        if (i >= numIterations - 1) {
            scan_block_kernel<<<grid_size, BLOCK_SIZE>>>(arrayPointers[i], arraySizes[i]);
            break;
        }

        cudaMallocAsync(&arrayPointers[i+1], grid_size*4, cudaStreamDefault);
        scan_block_kernel<<<grid_size, BLOCK_SIZE>>>(arrayPointers[i], arraySizes[i], arrayPointers[i+1]);
        arraySizes[i+1] = grid_size;
        grid_size = ceil(float(grid_size) / float(BLOCK_SIZE));
    }

    // Upsample block
    for (int i = numIterations-1; i > 0; --i) {
        add_scan_block_offsets_kernel<<<arraySizes[i], BLOCK_SIZE>>>(arrayPointers[i-1], arraySizes[i-1], arrayPointers[i]);
        cudaFreeAsync(arrayPointers[i], cudaStreamDefault);
    }

    // END GPU Implementation

    cudaMemcpy(B_h, XY_d, arraySize*4, cudaMemcpyDeviceToHost);
    cudaFree(XY_d);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < 10; ++i) {
        std::cout << "B_ref[" << i << "]: " << B_ref[i] << "\tB_h[" << i << "]: " << B_h[i] << std::endl;
    }

    // Compare GPU and CPU to look for errors
    for (size_t i = 0; i < arraySize; ++i) {
        if (!isSameFloat(B_h[i], B_ref[i])) {
            std::cout << "CPU and GPU results are not equal! " << B_h[i] << std::endl;
            std::cout << "B_ref[" << i << "]: " << B_ref[i] << "\tB_h[" << i << "]: " << B_h[i] << std::endl;
            return 1;
        }
    }

    std::cout << "Scan working as expected!" << std::endl;
    return 0;
}
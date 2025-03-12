#include <iostream>
#define NUM_BINS 7 // ceil(26/4) = 7

__global__ void histo_private_kernel(char* data, unsigned int length, unsigned int* histo) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[blockIdx.x*NUM_BINS + alphabet_position/4]), 1);
        }
    }
    if (blockIdx.x > 0) {
        __syncthreads();
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
            unsigned int binValue = histo[blockIdx.x*NUM_BINS + bin];
            if (binValue > 0) {
                atomicAdd(&(histo[bin]), binValue);
            }
        }
    }
}

__global__ void histo_private_kernel_shared_mem(char* data, unsigned int length, unsigned int* histo) {
    // initialize privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // Histogram
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }
    }
    __syncthreads();

    // Commit to global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&(histo[bin]), binValue);
        }
    }
}

int main() {
    std::string string_h = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et"
                           "dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex"
                           "ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu"
                           "fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt"
                           "mollit anim id est laborum";

    char* string_d;
    cudaMalloc(&string_d, string_h.length());
    cudaMemcpy(string_d, string_h.c_str(), string_h.length(), cudaMemcpyHostToDevice);

    unsigned int* hist_h = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
    unsigned int* hist_d;
    cudaMalloc(&hist_d, NUM_BINS*sizeof(unsigned int));

    dim3 blockDim = 128;
    dim3 gridDim = (string_h.length() + blockDim.x - 1) / blockDim.x;
    histo_private_kernel<<<gridDim, blockDim>>>(string_d, string_h.length(), hist_d);

    cudaMemcpy(hist_h, hist_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_BINS; i++) {
        std::cout << "Bucket " << i << ": " << hist_h[i] << std::endl;
    }

    cudaFree(string_d);
    cudaFree(hist_d);
    free(hist_h);
}
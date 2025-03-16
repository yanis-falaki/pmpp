#define BLOCK_DIM 1024

__global__ void SharedMemorySumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];

    // load initial iteration of reduction into shared memory
    input_s[threadIdx.x] = input[threadIdx.x] + input[threadIdx.x + BLOCK_DIM];
    
    for (unsigned int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x <= stride) {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0)
        *output = input_s[threadIdx.x];
}
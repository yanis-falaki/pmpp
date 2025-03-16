#define BLOCK_DIM 1024
__global__ void SegmentedSumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];

    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    input_s[t] = input[i] + input[i + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}


#define COARSE_FACTOR 3
__global__ void CoarsenedSumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];

    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float sum = input[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile) {
        sum += input[i + tile*BLOCK_DIM];
    }
    input_s[t] = sum;

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}
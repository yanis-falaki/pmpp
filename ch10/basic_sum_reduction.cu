__global__ void SimpleSumReductionKernel(float* input, float* output) {
    unsigned int i = 2*threadIdx.x;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride= blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

/*
Second approach improves resource utilization % as there are fewer active warps through time while the total amounts of
committed results have not changed.

Memory divergence also improves as in the second kernel each thread in a warp accesses adjacent memory locations and so read/writes can be coalesced.
*/
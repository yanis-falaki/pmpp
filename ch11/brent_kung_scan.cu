#define SECTION_SIZE 1024

__global__ void Brent_Kung_scan_kernel(float* X, float* Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) XY[threadIdx.x] = X[i];
    if (i + blockDim.x < N) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1)*2*stride - 1;
        if (index < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }
    for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1)*stride*2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    if (i < N) Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}
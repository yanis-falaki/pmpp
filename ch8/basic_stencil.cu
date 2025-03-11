__constant__ int c0 = 0;
__constant__ int c1 = 1;
__constant__ int c2 = 1;
__constant__ int c3 = 1;
__constant__ int c4 = 1;
__constant__ int c5 = 1;
__constant__ int c6 = 1;


__global__ void stencil_kernel(float* in, float* out, unsigned int N) {

    int i = blockIdx.z*blockDim.z + threadIdx.z;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        out[i*N*N + j*N + k] = c0*in[i*N*N + j*N + k]
                                + c1*in[i*N*N + j*N + (k-1)]
                                + c2*in[i*N*N + j*N + (k+1)]
                                + c3*in[i*N*N + (j-1)*N + k]
                                + c4*in[i*N*N + (j+1)*N + k]
                                + c5*in[(i-1)*N*N + j*N + k]
                                + c6*in[(i+1)*N*N + j*N + k];
    }
}
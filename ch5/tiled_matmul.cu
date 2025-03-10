#define TILE_WIDTH 16

#include <iostream>

__global__ void matrixMulKernel(float* M, float* N, float* P, int Width) {

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P elements
    float Pvalue = 0;
    for (int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ++ph) {

        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph*TILE_WIDTH+tx) < Width)
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        else Mds[ty][tx] = 0.0f;
        if ((ph*TILE_WIDTH+ty) < Width && Col < Width)
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        else Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }
    if ((Row < Width) && (Col < Width))
        P[Row*Width + Col] = Pvalue;
}

void CPUMatMul(float* A_h, float* B_h, float* C_h, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0;
            for (int k = 0; k < width; ++k)
                sum += A_h[i*width + k] * B_h[k * width + j];
            C_h[i * width + j] = sum;
        }
    }
}

void checkMats(float* A, float* B, int width) {
    bool isValid = true;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            if (A[i*width + j] != B[i*width + j]) {
                std::cout << i << ", " << j << " Indices are not equal." << std::endl;
                std::cout << "A value: " << A[i*width + j] << std::endl;
                std::cout << "B value: " << B[i*width + j] << std::endl;
                isValid = false;
            }
        }
    }

    if (isValid)
        std::cout << "Successful MatMul!" << std::endl;
}

int main() {
    int width = TILE_WIDTH * 6;

    float A_h[width*width];
    float B_h[width*width];
    float C_h[width*width];
    float C_ref[width*width];

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            A_h[i*width + j] = 1.0f; // Example values
            B_h[i*width + j] = 2.0f;
        }
    }

    float* A_d;
    float* B_d;
    float* C_d;

    cudaMalloc(&A_d, width*width*4);
    cudaMemcpy(A_d, A_h, width*width*4, cudaMemcpyHostToDevice);
    cudaMalloc(&B_d, width*width*4);
    cudaMemcpy(B_d, B_h, width*width*4, cudaMemcpyHostToDevice);
    cudaMalloc(&C_d, width*width*4);

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridSize(6, 6, 1);
    matrixMulKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, width);

    cudaMemcpy(C_h, C_d, width*width*4, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // compute on CPU
    CPUMatMul(A_h, B_h, C_ref, width);

    // verify
    checkMats(C_h, C_ref, width);

    return 0;
}
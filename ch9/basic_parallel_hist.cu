#include <iostream>

__global__ void histo_kernel(char* data, unsigned int length, unsigned int* histo) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[alphabet_position/4]), 1);
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

    unsigned int* hist_h = (unsigned int*)malloc(7*sizeof(unsigned int));
    unsigned int* hist_d;
    cudaMalloc(&hist_d, 7*sizeof(unsigned int));

    histo_kernel<<<1, string_h.length()>>>(string_d, string_h.length(), hist_d);

    cudaMemcpy(hist_h, hist_d, 7*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 7; i++) {
        std::cout << "Bucket " << i << ": " << hist_h[i] << std::endl;
    }

    cudaFree(string_d);
    cudaFree(hist_d);
    free(hist_h);
}
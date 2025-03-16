#include <iostream>

void sequential_scan(float* x, float* y, unsigned int N) {
    y[0] = x[0];
    for (unsigned int i = 1; i < N; ++i) {
        y[i] = y[i - 1] + x[i];
    }
}

int main() {
    float x_h[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float y_h[8];

    sequential_scan(x_h, y_h, 8);

    for (unsigned int i = 0; i < 8; ++i) {
        std::cout << y_h[i] << std::endl;
    }

    return 0;
}
#include <stdio.h>

__global__ void compute() {
    __shared__ float arr[32];
    int idx = threadIdx.x;

    arr[idx] = (float)idx;
    __syncthreads();

    if(idx > 0) {
        arr[idx-1] = arr[idx];
    }

    printf("arr[%d] = %f\n", idx, arr[idx]);
}

__global__ void compute_barrier() {
    __shared__ float arr[32];
    int idx = threadIdx.x;

    arr[idx] = (float)idx;
    __syncthreads();

    if(idx > 0) {
        float temp = arr[idx];
        __syncthreads();
        arr[idx-1] = temp;
        __syncthreads();
    }

    printf("arr[%d] = %f\n", idx, arr[idx]);
}

int main(int argc, char** argv) {
    int ARRAY_SIZE = 32;

    float in[ARRAY_SIZE];
    float out[ARRAY_SIZE];
    for(int i=0; i<ARRAY_SIZE; i++) {
        in[i] = (float)i;
        out[i] = in[i];
    }

    for(int i=0; i<ARRAY_SIZE; i++) {
        if(i > 0) {
            out[i-1] = in[i];
        }
    }
    printf("Reference:\n");
    for(int i=0; i<ARRAY_SIZE; i++) {
        printf("ref[%d] = %f\n", i, out[i]);
    }

    // Without Barrier - Incorrect
    printf("Kernel - Without Barrier:\n");
	compute<<<1, ARRAY_SIZE>>>();

    // With Barrier - Correct
    printf("Kernel - With Barrier:\n");
	compute_barrier<<<1, ARRAY_SIZE>>>();
	
	cudaDeviceSynchronize();
	
	return 0;
}

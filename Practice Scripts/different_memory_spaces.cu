#include <stdio.h>

// Using different memory spaces in CUDA

/***********
local memory
************/

// a __device__ or __global__ function runs on the GPU
__global__ void use_local_memory_GPU(float in) {
    float f;   // Variable "f" is in local memory and private to each thread
    f = in;    // Parameter "in" is in local memory and private to each thread

    //...      // Do computation here
}

__global__ void use_global_memory_GPU(float *in) {
    // "in" is a pointer to global memory on the device
    in[threadIdx.x] = 2.0f * (float) threadIdx.x;
}

__global__ void use_shared_memory_GPU(float *in) {
    // local variables private to each thread
    int i;
    int index = threadIdx.x;
    float average, sum = 0.0f;

    // shared variables are visible to all threads in the thread block
    // and have the same lifetime as the thread block
    __shared__ float sh_arr[128];

    // copy data from "in" in global memory to sh_arr in shared memory
    // each thread responsible for a single element here
    sh_arr[index] = in[index];

    __syncthreads();    // ensure all writes to shared memory are completed

    // sh_arr is fully populated. Find average of all previous elements
    for(i=0; i<index; i++) {
        sum += sh_arr[i];
    }
    average = sum / (index + 1.0f);

    // if in[index] is greater than the average of in[0...index-1], replace with average
    // since in[] is in global memory, this change will be seen by the host (an potentially other thread blocks, if any)
    if(in[index] > average) {
        in[index] = average;
    }

    // the following has NO EFFECT. It modifies shared memory, but the resulting modified data is never copied back
    // to global memory and vanishes when the thread block completes
    sh_arr[index] = 3.14;
}

int main(int argc, char **argv) {
    
    /*
     * Kernel using only local memory
     */
    use_local_memory_GPU<<<1, 128>>>(2.0f);

    /*
     * Kernel using global memory
     */
    float h_arr[128];
    float *d_arr;

    // allocate global memory on the device
    cudaMalloc((void**) &d_arr, sizeof(float)*128);
    // copy data from host memory "h_arr" to device memory "d_arr"
    cudaMemcpy(d_arr, h_arr, sizeof(float)*128, cudaMemcpyHostToDevice);

    // launch kernel - 1 block of 128 threads
    use_global_memory_GPU<<<1, 128>>>(d_arr);

    // copy data back to host, overwriting "h_arr"
    cudaMemcpy(h_arr, d_arr, sizeof(float)*128, cudaMemcpyDeviceToHost);

    /*
     * Kernel using shared memory
     */
    use_shared_memory_GPU<<<1, 128>>>(d_arr);
    // copy data back to host, overwriting "h_arr"
    cudaMemcpy(h_arr, d_arr, sizeof(float)*128, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    return 0;
}
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

int get_max_size(int n, int d) {
    return (int)ceil( (float)n/(float)d ) + 1;
}

__global__ void shmem_reduce_kernel(float *d_out, float *d_in) {
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = (blockDim.x * blockIdx.x) + threadIdx.x;
    int tid = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();

    // do reduction in shared memory
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if(tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// calculate reduce max or min and stick the value in d_answer.
__global__ void reduce_minmax_kernel(const float* const d_in, float* d_out, const size_t size, int minmax) {
    extern __shared__ float shared[];
    
    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x; 
    
    // we have 1 thread per block, so copying the entire block should work fine
    if(mid < size) {
        shared[tid] = d_in[mid];
    } else {
        if(minmax == 0)
            shared[tid] = FLT_MAX;
        else
            shared[tid] = -FLT_MAX;
    }
    
    // wait for all threads to copy the memory
    __syncthreads();
    
    // don't do any thing with memory if we happen to be far off ( I don't know how this works with
    // sync threads so I moved it after that point )
    if(mid >= size) {   
        if(tid == 0) {
            if(minmax == 0) 
                d_out[blockIdx.x] = FLT_MAX;
            else
                d_out[blockIdx.x] = -FLT_MAX;

        }
        return;
    }
       
    for(unsigned int s = blockDim.x/2; s > 0; s /= 2) {
        if(tid < s) {
            if(minmax == 0) {
                shared[tid] = min(shared[tid], shared[tid+s]);
            } else {
                shared[tid] = max(shared[tid], shared[tid+s]);
            }
        }
        
        __syncthreads();
    }
    
    if(tid == 0) {
        d_out[blockIdx.x] = shared[0];
    }
}

float reduce_minmax(const float* const d_in, const size_t size, int minmax) {
    int BLOCK_SIZE = 32;
    // we need to keep reducing until we get to the amount that we consider 
    // having the entire thing fit into one block size
    size_t curr_size = size;
    float* d_curr_in;
    
    cudaMalloc(&d_curr_in, sizeof(float) * size);    
    cudaMemcpy(d_curr_in, d_in, sizeof(float) * size, cudaMemcpyDeviceToDevice);


    float* d_curr_out;
    
    dim3 thread_dim(BLOCK_SIZE);
    const int shared_mem_size = sizeof(float)*BLOCK_SIZE;
    
    while(curr_size >= BLOCK_SIZE) {
        cudaMalloc(&d_curr_out, sizeof(float) * get_max_size(curr_size, BLOCK_SIZE));
        
        // Method 1
        dim3 block_dim(get_max_size(size, BLOCK_SIZE));
        reduce_minmax_kernel<<<block_dim, thread_dim, shared_mem_size>>>(
            d_curr_in,
            d_curr_out,
            curr_size,
            minmax
        );

        // Method 2
        // dim3 block_dim(get_max_size(curr_size, BLOCK_SIZE));
        // shmem_reduce_kernel<<<block_dim, thread_dim, shared_mem_size>>>(d_curr_out, d_curr_in);
        
        cudaDeviceSynchronize(); cudaGetLastError();

            
        // move the current input to the output, and clear the last input if necessary
        cudaFree(d_curr_in);
        d_curr_in = d_curr_out;
        
        // if(curr_size <  BLOCK_SIZE) 
        //     break;
        
        curr_size = get_max_size(curr_size, BLOCK_SIZE);
    }
    
    // theoretically we should be 
    float h_out;
    cudaMemcpy(&h_out, d_curr_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_curr_out);
    return h_out;
}

__global__ void simple_add(float *d_arr, float *d_orig) {
    int idx = threadIdx.x;

    d_arr[idx] += 1;
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0) {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %lldB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (long long int)devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate);
    }

    const int ARRAY_SIZE = 1 << 20;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float min_val = FLT_MAX;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)random()/((float)RAND_MAX/2.0f);
        min_val = min(min_val, h_in[i]);
    }

    // declare GPU memory pointers
    float * d_in;

    // allocate GPU memory
    cudaMalloc((void **)&d_in, ARRAY_BYTES);

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float h_out;
    // launch the kernel
    printf("Running reduce with shared mem\n");
    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; i++) {
        h_out = reduce_minmax(d_in, ARRAY_SIZE, 0);
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime /= 100.0f;      // 100 trials

    printf("average time elapsed: %f ms\n", elapsedTime);

    // free GPU memory allocation
    cudaFree(d_in);
    
    printf("CPU answer: %f\n", min_val);
    printf("GPU answer: %f\n", h_out);

    // Test memcpy
    float arr[5] = {0, 1.0, 1.1, 2.2, 3.3};
    float arr_print[5];
    float *d_arr, *d_arr2;

    for(int i=0; i<5; i++)
        printf("Arr original: %f\n", arr[i]);

    cudaMalloc((void **)&d_arr, sizeof(float)*5);
    cudaMalloc((void **)&d_arr2, sizeof(float)*5);
    cudaMemcpy(d_arr, arr, sizeof(float)*5, cudaMemcpyHostToDevice);

    cudaMemcpy(d_arr2, d_arr, sizeof(float)*5, cudaMemcpyDeviceToDevice);
    simple_add<<<1, 5>>>(d_arr2, d_arr);

    cudaMemcpy(&arr_print, d_arr2, sizeof(float)*5, cudaMemcpyDeviceToHost);
    for(int i=0; i<5; i++)
        printf("Arr2: %f\n", arr_print[i]);

    cudaMemcpy(&arr_print, d_arr, sizeof(float)*5, cudaMemcpyDeviceToHost);
    for(int i=0; i<5; i++)
        printf("Arr1: %f\n", arr_print[i]);
    return 0;
}
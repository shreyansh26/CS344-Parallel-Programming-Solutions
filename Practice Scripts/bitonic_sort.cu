#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gputimer.h"

// Your job is to implemment a bitonic sort. A description of the bitonic sort
// can be see at:
// http://en.wikipedia.org/wiki/Bitonic_sort

__device__
int powerOfTwo(const short N) {

    switch( N ) {
        case 0:
            return 1;
        case 1: 
            return 2;
        case 2:
            return 4;
        case 3:
            return 8;
        case 4: 
            return 16;
        case 5: 
            return 32;
        case 6: 
            return 64;
        default:
            return 1;
    }
}


__device__
void compareFloatAndSwap(float * data, const int x, const int y) {
  float temp = 0;
  if ( data[y] < data[x] ) {
      temp = data[x];
      data[x] = data[y];
      data[y] = temp;
  }
}


__global__ void batcherBitonicMergesort64(float * d_out, const float * d_in)
{
    // you are guaranteed this is called with <<<1, 64, 64*4>>>
    extern __shared__ float data[];
    int tid  = threadIdx.x;
    data[tid] = d_in[tid];
    __syncthreads();
    
    int _pow1, _pow2;
    for (int stage = 0; stage <= 5; stage++)
    {
        _pow1 = powerOfTwo(stage + 1);
        for (int substage = stage; substage >= 0; substage--)
        {    
           _pow2 = powerOfTwo(substage);
           if ( (tid/_pow1) % 2  ) {
                if( (tid/_pow2) % 2 ) compareFloatAndSwap(data, tid, tid - _pow2);
           } else {
                if( (tid/_pow2) % 2 == 0 ) compareFloatAndSwap(data, tid, tid + _pow2);
           }
           __syncthreads();
        }
    }
    d_out[tid] = data[tid];
}


int compareFloat (const void * a, const void * b)
{
  if ( *(float*)a <  *(float*)b ) return -1;
  if ( *(float*)a == *(float*)b ) return 0;
  if ( *(float*)a >  *(float*)b ) return 1;
  return 0;                     // should never reach this
}


int main(int argc, char **argv)
{
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float h_sorted[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [0, 1]
        h_in[i] = (float)random()/(float)RAND_MAX;
        h_sorted[i] = h_in[i];
    }
    qsort(h_sorted, ARRAY_SIZE, sizeof(float), compareFloat);

    // declare GPU memory pointers
    float * d_in, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    // launch the kernel
    GpuTimer timer;
    timer.Start();
    batcherBitonicMergesort64<<<1, ARRAY_SIZE, ARRAY_SIZE * sizeof(float)>>>(d_out, d_in);
    timer.Stop();
    
    printf("Your code executed in %g ms\n", timer.Elapsed());
    
    // copy back the sum from GPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    
    // compare your result against the reference solution
    int flag = 0;
    for(int i=0; i<ARRAY_SIZE; i++) {
        if(h_out[i] != h_sorted[i]) {
            flag = 1;
            printf("Error!\n");
        }
    }
    if(flag == 0)
        printf("PASS\n");
    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);
}
/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "timer.h"

__global__
void yourHisto_naive(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(idx >= numVals)
      return;

  atomicAdd(&histo[vals[idx]], 1);
}

__global__
void yourHisto_grid_stride_loop(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if(idx >= numVals)
      return;

  while(idx < numVals) {
    atomicAdd(&histo[vals[idx]], 1);
    idx += stride;
  }
}

__global__
void yourHisto_shmem_naive(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  extern __shared__ unsigned int local_histo[];
  local_histo[threadIdx.x] = 0;
  __syncthreads();

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(idx >= numVals)
      return;

  atomicAdd(&local_histo[vals[idx]], 1);
  __syncthreads();

  // Only 1024 global memory writes
  atomicAdd(&histo[threadIdx.x], local_histo[threadIdx.x]);
}

__global__
void yourHisto_shmem_grid_stride_loop(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  extern __shared__ unsigned int local_histo[];
  local_histo[threadIdx.x] = 0;
  __syncthreads();

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if(idx >= numVals)
      return;

  while(idx < numVals) {
    atomicAdd(&local_histo[vals[idx]], 1);
    idx += stride;
  }
  __syncthreads();

  // Only 1024 global memory writes
  atomicAdd(&histo[threadIdx.x], local_histo[threadIdx.x]);
}

int get_max_size(int n, int dim) {
  return (int)ceil((float) n / (float) dim) + 1;
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free
  dim3 thread_dim(numBins);
  dim3 block_dim(get_max_size(numElems, numBins));
  dim3 strided_block_dim(32*16);

  const int VAL_BYTES = numElems*sizeof(unsigned int);
  const int HISTO_BYTES = numBins*sizeof(unsigned int);
  GpuTimer timer;

  unsigned int *d_vals_t, *d_histo_t, *d_vals_original, *d_histo_original;
  cudaMalloc(&d_vals_t, VAL_BYTES);
  cudaMalloc(&d_histo_t, HISTO_BYTES);
  cudaMalloc(&d_vals_original, VAL_BYTES);
  cudaMalloc(&d_histo_original, HISTO_BYTES);

  cudaMemcpy(d_vals_original, d_vals, VAL_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_histo_original, d_histo, HISTO_BYTES, cudaMemcpyDeviceToDevice);

  cudaMemcpy(d_vals_t, d_vals_original, VAL_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_histo_t, d_histo_original, HISTO_BYTES, cudaMemcpyDeviceToDevice);
  timer.Start();
  yourHisto_naive<<<block_dim, thread_dim>>>(d_vals_t, d_histo_t, numElems);
  timer.Stop();
  printf("yourHisto_naive: %g ms.\n", timer.Elapsed());
  cudaMemcpy(d_histo, d_histo_t, HISTO_BYTES, cudaMemcpyDeviceToDevice);

  cudaMemcpy(d_vals_t, d_vals_original, VAL_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_histo_t, d_histo_original, HISTO_BYTES, cudaMemcpyDeviceToDevice);
  timer.Start();
  yourHisto_grid_stride_loop<<<strided_block_dim, thread_dim>>>(d_vals_t, d_histo_t, numElems);
  timer.Stop();
  printf("yourHisto_grid_stride_loop: %g ms.\n", timer.Elapsed());
  cudaMemcpy(d_histo, d_histo_t, HISTO_BYTES, cudaMemcpyDeviceToDevice);

  cudaMemcpy(d_vals_t, d_vals_original, VAL_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_histo_t, d_histo_original, HISTO_BYTES, cudaMemcpyDeviceToDevice);
  const int shared_mem_size = HISTO_BYTES;
  timer.Start();
  yourHisto_shmem_naive<<<block_dim, thread_dim, shared_mem_size>>>(d_vals_t, d_histo_t, numElems);
  timer.Stop();
  printf("yourHisto_shmem_naive: %g ms.\n", timer.Elapsed());
  cudaMemcpy(d_histo, d_histo_t, HISTO_BYTES, cudaMemcpyDeviceToDevice);

  cudaMemcpy(d_vals_t, d_vals_original, VAL_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_histo_t, d_histo_original, HISTO_BYTES, cudaMemcpyDeviceToDevice);
  // const int shared_mem_size = HISTO_BYTES;
  timer.Start();
  yourHisto_shmem_grid_stride_loop<<<strided_block_dim, thread_dim, shared_mem_size>>>(d_vals_t, d_histo_t, numElems);
  timer.Stop();
  printf("yourHisto_shmem_grid_stride_loop: %g ms.\n", timer.Elapsed());
  cudaMemcpy(d_histo, d_histo_t, HISTO_BYTES, cudaMemcpyDeviceToDevice);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

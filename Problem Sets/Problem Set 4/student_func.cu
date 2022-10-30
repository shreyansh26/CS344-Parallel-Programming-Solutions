//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include "debug.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void histogram_kernel(unsigned int* const d_in, unsigned int *d_hist, const size_t size, int lsb_idx) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(idx >= size)
    return;

  int binId = (d_in[idx] & (1U << lsb_idx)) == (1U << lsb_idx);
  // BinIds are 0 or 1
  atomicAdd(&d_hist[binId], 1);
}

__global__ void scan_exclusive_kernel(unsigned int *d_in, unsigned int *d_out, const size_t size, int lsb_idx, int block_idx, int threadSize) {
  // Do blockwise
  int idx = (threadSize * block_idx) + threadIdx.x;
  int block = threadSize * block_idx;

  if(idx >= size)
    return;

  unsigned int val = 0;

  if(idx > 0) {
    val = (d_in[idx-1] & (1U << lsb_idx)) == (1U << lsb_idx);
  }
  else {
    val = 0;
  }

  d_out[idx] = val;
  __syncthreads();

  for(int wid=1; wid<=size; wid *= 2) {
    int left = idx - wid;

    if(left >= 0 && left >= block) {
        val = d_out[left];
    }
    __syncthreads();
    if(left >= 0 && left >= block) {
        d_out[idx] += val;
    }
    __syncthreads();
  }

  // Add previous block scan elements
  if(block_idx > 0)
    d_out[idx] += d_out[block-1];
}

__global__ void move_elements_kernel(unsigned int *d_inputVals, unsigned int *d_inputPos, unsigned int *d_outputVals, 
                                      unsigned int *d_outputPos, unsigned int *d_newindex, unsigned int *d_scan, 
                                      unsigned int num_zeros, int lsb_idx, const size_t size) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(idx >= size)
    return;

  int pre_zeros = 0;
  int pre_ones = num_zeros;
  unsigned int fin_pos;

  if((d_inputVals[idx] & (1U << lsb_idx)) == (1U << lsb_idx)) {
    fin_pos = pre_ones + d_scan[idx];
  }
  else {
    fin_pos = pre_zeros + (idx - d_scan[idx]);
  }

  d_newindex[idx] = fin_pos;
  d_outputVals[fin_pos] = d_inputVals[idx];
  d_outputPos[fin_pos] = d_inputPos[idx];
}


int get_max_size(int n, int b) {
  return (int)ceil((float)n/(float)b) + 1;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
  int numBins = 2;

  unsigned int h_hist[numBins];
  unsigned int *d_hist;
  unsigned int *d_scan;
  unsigned int *d_newindex;

  const size_t hist_size = sizeof(unsigned int) * numBins;
  const size_t arr_size = sizeof(unsigned int) * numElems;
  checkCudaErrors(cudaMalloc((void **)&d_hist, hist_size));
  checkCudaErrors(cudaMalloc((void **)&d_scan, arr_size));
  checkCudaErrors(cudaMalloc((void **)&d_newindex, arr_size));

  int num_threads = 1024;
  dim3 thread_dim(num_threads);
  dim3 hist_block_dim(get_max_size(numElems, num_threads));

  for(int lsb_idx=0; lsb_idx<32; lsb_idx++) {
    checkCudaErrors(cudaMemset(d_hist, 0, hist_size));
    checkCudaErrors(cudaMemset(d_scan, 0, arr_size));
    checkCudaErrors(cudaMemset(d_outputVals, 0, arr_size));
    checkCudaErrors(cudaMemset(d_outputPos, 0, arr_size));
    checkCudaErrors(cudaMemset(d_newindex, 0, arr_size));

    histogram_kernel<<<hist_block_dim, thread_dim>>>(d_inputVals, d_hist, numElems, lsb_idx);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // copy the histogram data to host
    checkCudaErrors(cudaMemcpy(&h_hist, d_hist, hist_size, cudaMemcpyDeviceToHost));
    // printf("LSB Index: %d Histogram Data: %d %d %d %d \n", lsb_idx, h_hist[0], h_hist[1], h_hist[0]+h_hist[1], numElems);

    // Start scan
    for(int block_idx=0; block_idx < get_max_size(numElems, num_threads); block_idx++) {
      scan_exclusive_kernel<<<dim3(1), thread_dim>>>(d_inputVals, d_scan, numElems, lsb_idx, block_idx, num_threads);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    // printf("made it past scan calculation\n");
    // debug_device_array("input", 100, d_inputVals, numElems);
    // debug_device_array("scan", 100, d_scan, numElems);
    // verify_scan(d_inputVals, d_scan, numElems, lsb_idx);

    // Move elements and get new index
    move_elements_kernel<<<hist_block_dim, thread_dim>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_newindex, 
                                                          d_scan, h_hist[0], lsb_idx, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // printf("made it past move calculation \n");
    // debug_device_array("move", 100, d_newindex, numElems);
    // debug_device_array("output vals", 100, d_outputVals, numElems);
    // debug_device_array("output pos", 100, d_outputPos, numElems);

    // Now copy the "new" partially soted data based on current LSB to input for next iteration
    checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, arr_size, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, arr_size, cudaMemcpyDeviceToDevice));
  }

  // debug_device_array("output vals", 100000, d_outputVals, numElems);
  // debug_device_array("output pos", 100, d_outputPos, numElems);

  checkCudaErrors(cudaFree(d_newindex));
  checkCudaErrors(cudaFree(d_scan));
  checkCudaErrors(cudaFree(d_hist));
}

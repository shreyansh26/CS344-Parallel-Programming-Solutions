/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <float.h>

__global__ void reduce_minmax_kernel_v2(float *d_out, float *d_in, size_t size, bool min_flag) {
   // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
   extern __shared__ float sdata[];

   int myId = (blockDim.x * blockIdx.x) + threadIdx.x;
   int tid = threadIdx.x;

   // load shared mem from global mem
   if(myId < size) {
        sdata[tid] = d_in[myId];
    } else {
        if(min_flag)
            sdata[tid] = FLT_MAX;
        else
            sdata[tid] = -FLT_MAX;
    }
   __syncthreads();


   if(myId >= size) {   
        if(tid == 0) {
            if(min_flag) 
                d_out[blockIdx.x] = FLT_MAX;
            else
                d_out[blockIdx.x] = -FLT_MAX;

        }
        return;
    }

   // do reduction in shared memory
   for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
      if(tid < s) {
         if(min_flag)
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
         else
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
   }

   if(tid == 0) {
      d_out[blockIdx.x] = sdata[0];
   }
}

int get_max_size(int n, int dim) {
   return (int)ceil((float) n / (float) dim) + 1;
}

__global__ void reduce_minmax_kernel(float *d_out, float *d_in, bool min_flag) {
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
         if(min_flag)
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
         else
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
   }

   if(tid == 0) {
      d_out[blockIdx.x] = sdata[0];
   }
}

float reduce_minmax(const float* const d_in, size_t size, bool min_flag) {
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
   
   while(1) {
      cudaMalloc(&d_curr_out, sizeof(float) * get_max_size(curr_size, BLOCK_SIZE));
      
      // Method 1
      // dim3 block_dim(get_max_size(size, BLOCK_SIZE));
      // reduce_minmax_kernel_v2<<<block_dim, thread_dim, shared_mem_size>>>(
      //    d_curr_out,
      //    d_curr_in,
      //    curr_size,
      //    min_flag
      // );

      // Method 2
      dim3 reduce_block_dim(get_max_size(curr_size, BLOCK_SIZE));
      reduce_minmax_kernel<<<reduce_block_dim, thread_dim, shared_mem_size>>>(d_curr_out, d_curr_in, min_flag);
      
      cudaDeviceSynchronize(); cudaGetLastError();

      // move the current input to the output, and clear the last input if necessary
      cudaFree(d_curr_in);
      d_curr_in = d_curr_out;
      
      // Still do the above when curr_size < BLOCK_SIZE as that is the last step
      if(curr_size <  BLOCK_SIZE) 
         break;

      curr_size = get_max_size(curr_size, BLOCK_SIZE);
   }
   
   // return the min/max element which is in d_curr_out
   float h_minmax;
   cudaMemcpy(&h_minmax, d_curr_out, sizeof(float), cudaMemcpyDeviceToHost);
   cudaFree(d_curr_out);
   return h_minmax;
}

__global__ void histogram_kernel(const float* const d_logLuminance, int *d_hist, float min_logLum, float max_logLum, int size, const size_t numBins) {
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   if(idx >= size)
      return;

   int binId = ((d_logLuminance[idx] - min_logLum) / (max_logLum - min_logLum)) * numBins;

   atomicAdd(&d_hist[binId], 1);
}

__global__ void scan_exclusive_kernel(unsigned int *d_cdf, int size) {
   // Inclusive scan + Shift
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   if(idx >= size)
      return;

   for(int wid=1; wid<=size; wid *= 2) {
      int left = idx - wid;
      int val = 0;

      if(left >= 0) {
         val = d_cdf[left];
      }
      __syncthreads();
      if(left >= 0) {
         d_cdf[idx] += val;
      }
      __syncthreads();
   }

   // Shift elements to right
   if(idx < size) {
      int temp = d_cdf[idx];
      __syncthreads();
      d_cdf[idx+1] = temp;
      __syncthreads();
   }
   if(idx == 0)
      d_cdf[idx] = 0;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

   // Reduce operation for min and max
   size_t size = numRows * numCols;

   min_logLum = reduce_minmax(d_logLuminance, size, true);
   max_logLum = reduce_minmax(d_logLuminance, size, false);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   printf("Min :%f, Max: %f\n", min_logLum, max_logLum);
   float range = max_logLum - min_logLum;

   printf("Total bins: %d\n", numBins);

   // Histogram operation for histogram of luminance
   int *d_hist;
   size_t hist_size = sizeof(int) * numBins;
   checkCudaErrors(cudaMalloc((void **)&d_hist, hist_size));
   checkCudaErrors(cudaMemset(d_hist, 0, hist_size));

   dim3 thread_dim = numBins;
   dim3 histogram_block_dim = get_max_size(size, numBins);

   histogram_kernel<<<histogram_block_dim, thread_dim>>>(d_logLuminance, d_hist, min_logLum, max_logLum, size, numBins);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   // Scan (exclusive) for CDF calulation
   checkCudaErrors(cudaMemcpy(d_cdf, d_hist, hist_size, cudaMemcpyDeviceToDevice));

   dim3 scan_block_dim(get_max_size(numBins, numBins));
   scan_exclusive_kernel<<<scan_block_dim, thread_dim>>>(d_cdf, size);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   checkCudaErrors(cudaFree(d_hist));
}

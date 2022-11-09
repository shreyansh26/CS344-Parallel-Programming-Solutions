//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>

__device__ bool isSafe(int x, int y, const size_t numRows, const size_t numCols) {
   return ((x < numCols) && (y < numRows));
}

__device__ int get1DIdx(int x, int y, const size_t numCols) {
   return y*numCols + x;
}

__device__ bool isNotWhite(uchar4 image) {
   return !((image.x == 255) && (image.y == 255) && (image.z == 255));
}

// Compute the mask and mark interior and border regions of the mask
__global__ void computeMaskInteriorBorder(const uchar4* const d_sourceImage,
                                          int *d_interiorMask,
                                          int *d_borderMask,
                                          const size_t numRowsSource,
                                          const size_t numColsSource) 
{
   int absolute_image_position_x = (blockIdx.x * blockDim.x) + threadIdx.x;
   int absolute_image_position_y = (blockIdx.y * blockDim.y) + threadIdx.y;
   int idx = get1DIdx(absolute_image_position_x, absolute_image_position_y, numColsSource);

   if(!isSafe(absolute_image_position_x, absolute_image_position_y, numRowsSource, numColsSource))
      return;

   if(isNotWhite(d_sourceImage[idx])) {
      int inbounds = 0;
      int interior = 0;

      for(int x=-1; x<=1; x++) {
         if(x == 0)
            continue;
         if(isSafe(absolute_image_position_x+x, absolute_image_position_y, numRowsSource, numColsSource)) {
            inbounds++;
            if(isNotWhite(d_sourceImage[get1DIdx(absolute_image_position_x+x, absolute_image_position_y, numColsSource)]))
               interior++;
         }
      }

      for(int y=-1; y<=1; y++) {
         if(y == 0)
            continue;
         if(isSafe(absolute_image_position_x, absolute_image_position_y+y, numRowsSource, numColsSource)) {
            inbounds++;
            if(isNotWhite(d_sourceImage[get1DIdx(absolute_image_position_x, absolute_image_position_y+y, numColsSource)]))
               interior++;
         }
      }

      d_interiorMask[idx] = 0;
      d_borderMask[idx] = 0;

      // Completely interior
      if(inbounds == interior)
         d_interiorMask[idx] = 1;
      // Atleast one neighbor inside
      else if(interior > 0)
         d_borderMask[idx] = 1;
   }
}

// The jacobi method
__global__ void jacobi(float *d_in, float *d_out, 
                        int *d_interiorMask, 
                        int *d_borderMask,
                        float *d_sourceChannel,
                        float *d_destChannel,
                        const size_t numRowsSource,
                        const size_t numColsSource) 
{
   int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                          blockIdx.y * blockDim.y + threadIdx.y);
   int idx = get1DIdx(thread_2D_pos.x, thread_2D_pos.y, numColsSource);

   if(!isSafe(thread_2D_pos.x, thread_2D_pos.y, numRowsSource, numColsSource))
      return;

   // Calculate A, B, C, D as described
   int neighbor_idx;
   if(d_interiorMask[idx] == 1) {
      float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;

      for(int x=-1; x<=1; x++) {
         if(x == 0)
            continue;
         if(isSafe(thread_2D_pos.x+x, thread_2D_pos.y, numRowsSource, numColsSource)) {
            d++;
            neighbor_idx = get1DIdx(thread_2D_pos.x+x, thread_2D_pos.y, numColsSource);
            if(d_interiorMask[neighbor_idx] == 1)
               a += d_in[neighbor_idx];
            else if(d_borderMask[neighbor_idx] == 1)
               b += d_destChannel[neighbor_idx];
            
            c += (d_sourceChannel[idx] - d_sourceChannel[neighbor_idx]);
         }
      }

      for(int y=-1; y<=1; y++) {
         if(y == 0)
            continue;
         if(isSafe(thread_2D_pos.x, thread_2D_pos.y+y, numRowsSource, numColsSource)) {
            d++;
            neighbor_idx = get1DIdx(thread_2D_pos.x, thread_2D_pos.y+y, numColsSource);
            if(d_interiorMask[neighbor_idx] == 1)
               a += d_in[neighbor_idx];
            else if(d_borderMask[neighbor_idx] == 1)
               b += d_destChannel[neighbor_idx];
            
            c += (d_sourceChannel[idx] - d_sourceChannel[neighbor_idx]);
         }
      }

      d_out[idx] = min(255.0f, max(0.0f, (a + b + c)/d));
   }
   else {
      d_out[idx] = d_destChannel[idx];
   }
}


//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4* const inputImageRGBA,
                                 int numRows,
                                 int numCols,
                                 float* const redChannel,
                                 float* const greenChannel,
                                 float* const blueChannel)
{
   int absolute_image_position_x = (blockIdx.x * blockDim.x) + threadIdx.x;
   int absolute_image_position_y = (blockIdx.y * blockDim.y) + threadIdx.y;

   if((absolute_image_position_x >= numCols) || (absolute_image_position_y >= numRows))
      return; 

   int idx = absolute_image_position_y * numCols + absolute_image_position_x;

   redChannel[idx] = (float)inputImageRGBA[idx].x;
   greenChannel[idx] = (float)inputImageRGBA[idx].y;
   blueChannel[idx] = (float)inputImageRGBA[idx].z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__ void recombineChannels(float* const redChannel,
                                 float* const greenChannel,
                                 float* const blueChannel,
                                 uchar4* const outputImageRGBA,
                                 int numRows,
                                 int numCols)
{
   const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                          blockIdx.y * blockDim.y + threadIdx.y);

   const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

   //make sure we don't try and access memory outside the image
   //by having any threads mapped there return early
   if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
      return;

   unsigned char red   = (unsigned char)redChannel[thread_1D_pos];
   unsigned char green = (unsigned char)greenChannel[thread_1D_pos];
   unsigned char blue  = (unsigned char)blueChannel[thread_1D_pos];

   //Alpha should be 255 for no transparency
   uchar4 outputPixel = make_uchar4(red, green, blue, 255);

   outputImageRGBA[thread_1D_pos] = outputPixel;
}

int get_max_size(int n, int d) {
    return (int)ceil( (float)n/(float)d ) + 1;
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
   const size_t imageSize = numRowsSource * numColsSource * sizeof(uchar4);

   uchar4* d_sourceImg;
	uchar4* d_destImg;
	uchar4* d_finalImg;

   checkCudaErrors(cudaMalloc(&d_sourceImg, imageSize));
   checkCudaErrors(cudaMalloc(&d_destImg, imageSize));
   checkCudaErrors(cudaMalloc(&d_finalImg, imageSize));

   checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, imageSize, cudaMemcpyHostToDevice));
  	checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, imageSize, cudaMemcpyHostToDevice));

	const size_t maskSize = numRowsSource*numColsSource*sizeof(int);
	int* d_borderMask;
	int* d_interiorMask;

   checkCudaErrors(cudaMalloc(&d_borderMask, maskSize));
	checkCudaErrors(cudaMalloc(&d_interiorMask, maskSize));

   const dim3 thread_dim(32, 32);
   const dim3 block_dim(get_max_size(numColsSource, thread_dim.x), get_max_size(numRowsSource, thread_dim.y));


   /* To Recap here are the steps you need to implement
  
   1) Compute a mask of the pixels from the source image to be copied
      The pixels that shouldn't be copied are completely white, they
      have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

   2) Compute the interior and border regions of the mask.  An interior
      pixel has all 4 neighbors also inside the mask.  A border pixel is
      in the mask itself, but has at least one neighbor that isn't.
   */
   computeMaskInteriorBorder<<<block_dim, thread_dim>>>(d_sourceImg, d_interiorMask, d_borderMask, numRowsSource, numColsSource);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   /*
   3) Separate out the incoming image into three separate channels
   */
   size_t floatSize = numRowsSource * numColsSource * sizeof(float);
	float *d_sourceImgR, *d_sourceImgG, *d_sourceImgB; 
	float *d_destImgR, *d_destImgG, *d_destImgB;

	checkCudaErrors(cudaMalloc(&d_sourceImgR, floatSize));
	checkCudaErrors(cudaMalloc(&d_sourceImgG, floatSize));
	checkCudaErrors(cudaMalloc(&d_sourceImgB, floatSize));
	
	checkCudaErrors(cudaMalloc(&d_destImgR, floatSize));
	checkCudaErrors(cudaMalloc(&d_destImgG, floatSize));
	checkCudaErrors(cudaMalloc(&d_destImgB, floatSize));

   separateChannels<<<block_dim, thread_dim>>>(d_sourceImg, numRowsSource, numColsSource, d_sourceImgR, d_sourceImgG, d_sourceImgB);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   separateChannels<<<block_dim, thread_dim>>>(d_destImg, numRowsSource, numColsSource, d_destImgR, d_destImgG, d_destImgB);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   /*
   4) Create two float(!) buffers for each color channel that will
      act as our guesses.  Initialize them to the respective color
      channel of the source image since that will act as our intial guess.
   */
   // allocate floats
	float *d_r0, *d_r1, *d_g0, *d_g1, *d_b0, *d_b1; 
	checkCudaErrors(cudaMalloc(&d_r0, floatSize));
	checkCudaErrors(cudaMalloc(&d_r1, floatSize));
	checkCudaErrors(cudaMalloc(&d_g0, floatSize));
	checkCudaErrors(cudaMalloc(&d_g1, floatSize));
   checkCudaErrors(cudaMalloc(&d_b0, floatSize));
	checkCudaErrors(cudaMalloc(&d_b1, floatSize));

  	checkCudaErrors(cudaMemcpy(d_r0, d_sourceImgR, floatSize, cudaMemcpyDeviceToDevice));
  	checkCudaErrors(cudaMemcpy(d_g0, d_sourceImgG, floatSize, cudaMemcpyDeviceToDevice));
  	checkCudaErrors(cudaMemcpy(d_b0, d_sourceImgB, floatSize, cudaMemcpyDeviceToDevice));

   /*
   5) For each color channel perform the Jacobi iteration described 
      above 800 times.
   */
   for(int iteration=0; iteration<800; iteration++) {
      jacobi<<<block_dim, thread_dim>>>(d_r0, d_r1, d_interiorMask, d_borderMask, d_sourceImgR, d_destImgR, numRowsSource, numColsSource);
      std::swap(d_r0, d_r1);

      jacobi<<<block_dim, thread_dim>>>(d_g0, d_g1, d_interiorMask, d_borderMask, d_sourceImgG, d_destImgG, numRowsSource, numColsSource);
      std::swap(d_g0, d_g1);

      jacobi<<<block_dim, thread_dim>>>(d_b0, d_b1, d_interiorMask, d_borderMask, d_sourceImgB, d_destImgB, numRowsSource, numColsSource);
      std::swap(d_b0, d_b1);
   }
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   /*
   6) Create the output image by replacing all the interior pixels
      in the destination image with the result of the Jacobi iterations.
      Just cast the floating point values to unsigned chars since we have
      already made sure to clamp them to the correct range.
   
   Since this is final assignment we provide little boilerplate code to
   help you.  Notice that all the input/output pointers are HOST pointers.

   You will have to allocate all of your own GPU memory and perform your own
   memcopies to get data in and out of the GPU memory.

   Remember to wrap all of your calls with checkCudaErrors() to catch any
   thing that might go wrong.  After each kernel call do:

   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   to catch any errors that happened while executing the kernel.
  */
  recombineChannels<<<block_dim, thread_dim>>>(d_r0, d_g0, d_b0, d_finalImg, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   // copy device final image to host
  	checkCudaErrors(cudaMemcpy(h_blendedImg, d_finalImg, imageSize, cudaMemcpyDeviceToHost));

	// cleanup
  	checkCudaErrors(cudaFree(d_sourceImg));
  	checkCudaErrors(cudaFree(d_destImg));
	checkCudaErrors(cudaFree(d_finalImg));

	checkCudaErrors(cudaFree(d_sourceImgR));
	checkCudaErrors(cudaFree(d_sourceImgG));
	checkCudaErrors(cudaFree(d_sourceImgB));

	checkCudaErrors(cudaFree(d_destImgR));
	checkCudaErrors(cudaFree(d_destImgG));
	checkCudaErrors(cudaFree(d_destImgB));

	checkCudaErrors(cudaFree(d_r0));
	checkCudaErrors(cudaFree(d_r1));
	checkCudaErrors(cudaFree(d_g0));
	checkCudaErrors(cudaFree(d_g1));
	checkCudaErrors(cudaFree(d_b0));
	checkCudaErrors(cudaFree(d_b1));
}

/*
 filename: square.cu
 compile:  nvcc -o square square.cu
*/

#include <stdio.h>

/*
	Device (GPU) code
	__global__ - declaration specifier - GPU code mark
*/
__global__ void square(float *d_out, float *d_in) {
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f;
}

/*
	Host (CPU) code
*/
int main(int argc, char** argv) {
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for( int i=0; i<ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}
	
	float h_out[ARRAY_SIZE];
	
	// declare GPU memory pointers
	float* d_in;
	float* d_out;
	
	// allocate GPU memory
	cudaMalloc( (void**) &d_in, ARRAY_BYTES);
	cudaMalloc( (void**) &d_out, ARRAY_BYTES);
	
	// transfer the array to the GPU
	// 4th parameter
	// cudaMemcpyDeviceToHost
	// cudaMemcpyDeviceToDevice
	// cudaMemcpyHostToDevice
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	
	// launch the kernel
	// 1 - launch on one block - can run many blocks at one
	/*
	maximum number of threads/block is 512 (older GPUs) or 1024 (newer GPUs)
	for instance:
		128 threads? 
			square<<<1,  '128>>>( ... )	OK
		1280 threads? 
			square<<<10, 128>>>( ... )	OK
			square<<<5,  256>>>( ... )	OK
			square<<<1, 1280>>>( ... ) !!!NO!!!
	For 2-dimensional problem, like image processing, use <<<128, 128>>>
	on a 128x128 pixels image.
	
	kernel<<< grid of blocks , block of threads >>> ( ... )
				1,2 or 3D       1,2 or 3D
	dim3( x, y, z )
	dim3( x, 1, 1 ) == dim3( w ) == w
	
	kernel<<< dim3( bx, by, bz ) , dim3( tx, ty, tz ), shmem >>> ( ... )
	
	bx * by * bz	grid of blocks
	tx * ty * tz	block of threads
	shmem			memory shared per block (in bytes)
	
	each thread knows it's threadId
	
	threadId	thread within a block
		threadId.x
		threadId.y
	blockDim	size of a block
	blockIdx	block within a grid
	gridDim		size of a grid
	
	
	Map concept
	
	Map:
		- set of elements (eg. 64 float array)
		- function to run on each element indipendently	(eg. square function)
	so Map is:
		map(elements,function)
	
	GPUs are good at map because:
		- gpus have many parallel processors
		- gpus optimize for throughput
	
	*/
	// ARRAY_SIZE - 64 copy of the kernel
	// pass the GPU data
	square<<<1, ARRAY_SIZE>>>(d_out, d_in);
	
	cudaDeviceSynchronize();
	
	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	// print out the resulting array
	for( int i=0; i<ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i%4) != 3) ? "\t" : "\n");
	}
	
	// free the GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_out);
	
	return 0;
}

#include <stdio.h>
#include "gputimer.h"

const int N = 1024;		// matrix size is NxN
const int K = 32;		// tile size is KxK

// Utility functions: compare, print, and fill matrices
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at: %s : %d\n", file,line);
    fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);;
    exit(1);
  }
}

int compare_matrices(float *gpu, float *ref)
{
	int result = 0;

	for(int j=0; j < N; j++)
    	for(int i=0; i < N; i++)
    		if (ref[i + j*N] != gpu[i + j*N])
    		{
    			// printf("reference(%d,%d) = %f but test(%d,%d) = %f\n",
    			// i,j,ref[i+j*N],i,j,test[i+j*N]);
    			result = 1;
    		}
    return result;
}

void print_matrix(float *mat)
{
	for(int j=0; j < N; j++) 
	{
		for(int i=0; i < N; i++) { printf("%4.4g ", mat[i + j*N]); }
		printf("\n");
	}	
}

// fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float *mat)
{
	for(int j=0; j < N * N; j++)
		mat[j] = (float) j;
}

void transpose_CPU(float in[], float out[]) {
    for(int j=0; j<N; j++) {
        for(int i=0; i<N; i++) {
            out[j + i*N] = in[i + j*N];  // out(j,i) = in(i,j)
        }
    }
}

// Launch using only one thread
__global__ void transpose_serial(float in[], float out[]) {
    for(int j=0; j<N; j++) {
        for(int i=0; i<N; i++) {
            out[j + i*N] = in[i + j*N];
        }
    }
}

// Launch with one thread for each row of output matrix
__global__ void transpose_parallel_per_row(float in[], float out[]) {
    int idx = threadIdx.x;

    for(int j=0; j<N; j++) {
        out[j + idx*N] = in[idx + j*N];
    }
}

// Launch with one thread for each element of output matrix. 
// Thread (x, y) writes to (i, j)
__global__ void transpose_parallel_per_element(float in[], float out[]) {
    int idxX = (blockIdx.x * blockDim.x + threadIdx.x);
    int idxY = (blockIdx.y * blockDim.y + threadIdx.y);

    out[idxY + idxX*N] = in[idxX + idxY*N];
}

// Launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
__global__ void transpose_parallel_per_element_tiled(float in[], float out[]) {
    __shared__ float tile[K][K];

    int baseX = (blockIdx.x * blockDim.x);
    int baseY = (blockIdx.y * blockDim.y);
    int x = threadIdx.x;
    int y = threadIdx.y;

    // // coalesced read from global mem, TRANSPOSED write into shared mem
    tile[y][x] = in[(baseX + x) + (baseY + y)*N];
    __syncthreads();

    // read from shared mem, coalesced write to global mem
    out[(baseY + x) + (baseX + y)*N] = tile[x][y];
}

// Launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
__global__ void transpose_parallel_per_element_tiled16(float in[], float out[]) {
    __shared__ float tile[16][16];

    int baseX = (blockIdx.x * 16);
    int baseY = (blockIdx.y * 16);
    int x = threadIdx.x;
    int y = threadIdx.y;

    // // coalesced read from global mem, TRANSPOSED write into shared mem
    tile[y][x] = in[(baseX + x) + (baseY + y)*N];
    __syncthreads();

    // read from shared mem, coalesced write to global mem
    out[(baseY + x) + (baseX + y)*N] = tile[x][y];
}

// Launched with one thread per element, in KxK threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array padded to avoid bank conflicts
__global__ void transpose_parallel_per_element_tiled_padded(float in[], float out[]) {
    __shared__ float tile[K][K+1];

    int baseX = (blockIdx.x * blockDim.x);
    int baseY = (blockIdx.y * blockDim.y);
    int x = threadIdx.x;
    int y = threadIdx.y;

    // // coalesced read from global mem, TRANSPOSED write into shared mem
    tile[y][x] = in[(baseX + x) + (baseY + y)*N];
    __syncthreads();

    // read from shared mem, coalesced write to global mem
    out[(baseY + x) + (baseX + y)*N] = tile[x][y];
}


// Launched with one thread per element, in KxK threadblocks
// thread blocks read & write tiles, in coalesced fashion
// shared memory array padded to avoid bank conflicts
__global__ void transpose_parallel_per_element_tiled_padded16(float in[], float out[]) {
    __shared__ float tile[16][16+1];

    int baseX = (blockIdx.x * 16);
    int baseY = (blockIdx.y * 16);
    int x = threadIdx.x;
    int y = threadIdx.y;

    // // coalesced read from global mem, TRANSPOSED write into shared mem
    tile[y][x] = in[(baseX + x) + (baseY + y)*N];
    __syncthreads();

    // read from shared mem, coalesced write to global mem
    out[(baseY + x) + (baseX + y)*N] = tile[x][y];
}

int main(int argc, char **argv) {
    int numBytes = N * N * sizeof(float);

    float *in = (float *)malloc(numBytes);
    float *out = (float *)malloc(numBytes);
    float *gold_out = (float *)malloc(numBytes);

    fill_matrix(in);
    transpose_CPU(in, gold_out);

    float *d_in, *d_out;

    cudaMalloc(&d_in, numBytes);
    cudaMalloc(&d_out, numBytes);

    cudaMemcpy(d_in, in, numBytes, cudaMemcpyHostToDevice);

    GpuTimer timer;

    /*  
    * Now time each kernel and verify that it produces the correct result.
    *
    * To be really careful about benchmarking purposes, we should run every kernel once
    * to "warm" the system and avoid any compilation or code-caching effects, then run 
    * every kernel 10 or 100 times and average the timings to smooth out any variance. 
    * But this makes for messy code and our goal is teaching, not detailed benchmarking.
    */

    timer.Start();
    transpose_serial<<<1, 1>>>(d_in, d_out);
    timer.Stop();
    cudaMemcpy(out, d_out, numBytes, cudaMemcpyDeviceToHost);
    printf("transpose_serial: %g ms.\nVerifying transpose...%s\n", 
	       timer.Elapsed(), compare_matrices(out, gold_out) ? "Failed" : "Success");

    timer.Start();
	transpose_parallel_per_row<<<1, N>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_row: %g ms.\nVerifying transpose...%s\n", 
		   timer.Elapsed(), compare_matrices(out, gold_out) ? "Failed" : "Success");

    dim3 thread_dim(K, K);
    dim3 block_dim(N/K, N/K);

    timer.Start();
	transpose_parallel_per_element<<<block_dim, thread_dim>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element: %g ms.\nVerifying transpose...%s\n", 
		   timer.Elapsed(), compare_matrices(out, gold_out) ? "Failed" : "Success");

    timer.Start();
	transpose_parallel_per_element_tiled<<<block_dim, thread_dim>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled: %g ms.\nVerifying transpose...%s\n", 
		   timer.Elapsed(), compare_matrices(out, gold_out) ? "Failed" : "Success");

    timer.Start();
	transpose_parallel_per_element_tiled_padded<<<block_dim, thread_dim>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled_padded: %g ms.\nVerifying transpose...%s\n", 
		   timer.Elapsed(), compare_matrices(out, gold_out) ? "Failed" : "Success");

    dim3 thread_dim16(16, 16);
    dim3 block_dim16(N/16, N/16);

    timer.Start();
	transpose_parallel_per_element_tiled16<<<block_dim16, thread_dim16>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled 16x16: %g ms.\nVerifying transpose...%s\n", 
		   timer.Elapsed(), compare_matrices(out, gold_out) ? "Failed" : "Success");

    timer.Start();
	transpose_parallel_per_element_tiled_padded16<<<block_dim16, thread_dim16>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numBytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled_padded 16x16: %g ms.\nVerifying transpose...%s\n", 
		   timer.Elapsed(), compare_matrices(out, gold_out) ? "Failed" : "Success");

    cudaFree(d_in);
    cudaFree(d_out);
}

/*
transpose_serial: 76.0254 ms.
Verifying transpose...Success
transpose_parallel_per_row: 1.56058 ms.
Verifying transpose...Success
transpose_parallel_per_element: 0.289984 ms.
Verifying transpose...Success
transpose_parallel_per_element_tiled: 0.20912 ms.
Verifying transpose...Success
transpose_parallel_per_element_tiled_padded: 0.185088 ms.
Verifying transpose...Success
transpose_parallel_per_element_tiled 16x16: 0.1944 ms.
Verifying transpose...Success
transpose_parallel_per_element_tiled_padded 16x16: 0.192704 ms.
Verifying transpose...Success
*/
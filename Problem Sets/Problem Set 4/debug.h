//Debugging
int debug = 1;
void debug_device_array(char* name, int l, unsigned int * d_arr, int numElems) {
    if(!debug)
        return;
    unsigned int h_arr[l];
    checkCudaErrors(cudaMemcpy(&h_arr, d_arr, l*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf(name);
    printf(" ");
    for(int i=0; i < l; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");
    unsigned int max = 0;
    unsigned int min = 1000000;
    unsigned int h_arr2[numElems];
    checkCudaErrors(cudaMemcpy(&h_arr2, d_arr, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    for(int i = 0; i < numElems; i++) {
        if(h_arr2[i] < min)
            min = h_arr2[i];
         if(h_arr2[i] > max)
            max = h_arr2[i];
    }
    printf("max %d min %d\n", max, min);
}

void verify_scan(unsigned int * d_arr, unsigned int * d_scanned, int numElems, int pass) {
    unsigned int h_arr[3000];
    unsigned int one  =1;
    unsigned int h_scanned[3000];
    checkCudaErrors(cudaMemcpy(&h_arr, d_arr, 3000*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_scanned, d_scanned, 3000*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int acc = 0;
    for(int i = 0; i < 3000; i++) {
        if(acc != h_scanned[i]) {
               printf("wrong at %d %d != %d\n", i, acc, h_scanned[i]);
        }
        acc += ((h_arr[i] & (one<<pass)) == (one<<pass)) ? 1 : 0;
    }
}
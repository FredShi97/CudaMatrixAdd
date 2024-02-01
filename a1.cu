#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>


#define BLOCK_SIZE_ROW 16
#define BLOCK_SIZE_COL 16

double getTimeStamp() {
    struct timeval  tv ; gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}


__global__ void f_siggen(float *d_x, float *d_y, float *d_z, int row_size, int col_size){
    
    //add one row above and one row below to accommendate for row - 1 and row + 1 read
    __shared__ float x_shared[(BLOCK_SIZE_ROW + 2) * BLOCK_SIZE_COL]; 
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int offset; 
    int shared_index;

    if (x < col_size && y < row_size){
        offset = y * col_size + x;  
        shared_index = (threadIdx.y + 1) * blockDim.x + threadIdx.x; 
        //copy out d_x based on block index
        x_shared[shared_index] = d_x[offset]; 
        //if its first row, copy out one row above from d_x
        if (threadIdx.y == 0 && (y - 1 >= 0))
            x_shared[threadIdx.x] = d_x[(y - 1) * col_size + x]; 
        //if its last row, copy out one row below from d_x
        if ((threadIdx.y == blockDim.y - 1) && (y + 1 < row_size))
            x_shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x] = d_x[(y + 1) * col_size + x];
    }
    
     __syncthreads(); 

    if (x < col_size && y < row_size){
        float output = x_shared[shared_index] - d_y[offset]; 

        //coalesced access. 
        if (x - 1 >= 0) 
            output -= d_y[y * col_size + x - 1];
        if (x - 2 >= 0)
            output -= d_y[y * col_size + x - 2]; 

        // read from shared memory 
        if (y - 1 >= 0)
            output += x_shared[threadIdx.y * blockDim.x + threadIdx.x];
        if (y + 1 < row_size)
            output +=  x_shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];

        d_z[offset] = output; 
    }
    

    
}

int main(int argc, char **argv) {

    if (argc != 3){
        printf("Program takes 3 args, <row_size> <col_size>"); 
        return 0; 
    }

    int row_size = atoi(argv[1]);
    int col_size = atoi(argv[2]); 

    float *h_x, *h_y, *h_z, *d_x, *d_y, *d_z; 
    int n_size = row_size * col_size;

    cudaHostAlloc((void**) &h_x, n_size * sizeof(float), 0); 
    cudaHostAlloc((void**) &h_y, n_size * sizeof(float), 0); 
    cudaHostAlloc((void**) &h_z, n_size * sizeof(float), 0); 
    cudaMalloc((void**) &d_x, n_size * sizeof(float));
    cudaMalloc((void**) &d_y, n_size * sizeof(float));
    cudaMalloc((void**) &d_z, n_size * sizeof(float));

    for (int i = 0; i < row_size; i++){
        for (int j = 0; j < col_size; j++){
            int offset = i * col_size + j; 
            h_x[offset] = (float) ((i+j) % 100) / 2.0; 
            h_y[offset] = (float) 3.25 * ((i+j) % 100); 
        }
    }



    dim3 blockSize(BLOCK_SIZE_COL, BLOCK_SIZE_ROW); 
    int gridX = (col_size + blockSize.x - 1) / blockSize.x;
    int gridY = (row_size + blockSize.y - 1) / blockSize.y;
    dim3 gridSize(gridX, gridY); 


    double startTime = getTimeStamp(); 
    double endTime; 
    double totalStartTime = startTime; 

    cudaMemcpy(d_x, h_x, n_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n_size * sizeof(float), cudaMemcpyHostToDevice);

    endTime = getTimeStamp();
    double CPU_GPU_Transfer_time = endTime - startTime; 
    startTime = endTime;

    f_siggen<<<gridSize,blockSize>>>(d_x, d_y, d_z, row_size, col_size); 
    cudaDeviceSynchronize(); 
    endTime = getTimeStamp();
    double kernel_time = endTime - startTime; 
    startTime = endTime;

    cudaMemcpy(h_z, d_z, n_size * sizeof(float), cudaMemcpyDeviceToHost);
    endTime = getTimeStamp();
    double GPU_CPU_Transfer_time = endTime - startTime; 

    double total_time = endTime - totalStartTime; 


    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z); 


    printf("%f %f %f %f %f \n", total_time, CPU_GPU_Transfer_time, kernel_time, 
    GPU_CPU_Transfer_time, h_z[5*col_size + 5]);

    
    for (int i = 0; i < row_size; i++){
        for (int j = 0; j < col_size; j++){
            int offset = i * col_size + j; 
            float out = 0.0;
            out = h_x[offset] - h_y[offset];
            if (i - 1 >= 0)
                out += h_x[(i - 1) * col_size + j];
            if (i + 1 < row_size)
                out += h_x[(i + 1) * col_size + j];
            if (j - 1 >= 0)
                out -= h_y[i * col_size + j - 1];
            if (j - 2 >= 0)
                out -= h_y[i * col_size + j - 2];   

            //printf("X is %f, Y is %f, row %d, col %d \n", h_x[offset], h_y[offset], i, j);
            
            if (h_z[offset] != out)
                printf("CPU calculated is %f, GPU is %f, row %d, col %d \n", out, h_z[offset], i, j);
        }
    }

    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFreeHost(h_z);
  

    return 0;
}


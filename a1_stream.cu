#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>


#define BLOCK_SIZE_ROW 16
#define BLOCK_SIZE_COL 16

#define N_STREAMS 2

double getTimeStamp() {
    struct timeval  tv ; gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}

void recordTimeCallback(cudaStream_t stream, cudaError_t status, void *data){
    *((double *)data)  = getTimeStamp();
}

__global__ void f_siggen(float *d_x, float *d_y, float *d_z, int row_size, int col_size){
    
    //add one row above and one row below to accommendate for row - 1 and row + 1 read
    __shared__ float x_shared[(BLOCK_SIZE_ROW + 2) * BLOCK_SIZE_COL]; 
    __shared__ float y_shared[BLOCK_SIZE_ROW * (BLOCK_SIZE_COL + 2)]; 
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int offset; 
    int shared_index_X;
    int shared_index_Y; 


    if (x < col_size && y < row_size ){

        offset = y * col_size + x;
        shared_index_Y = threadIdx.y * (blockDim.x + 2) + threadIdx.x + 2; 
        shared_index_X = (threadIdx.y + 1) * blockDim.x + threadIdx.x; 
        //copy out d_x based on block index
        x_shared[shared_index_X] = d_x[offset]; 
        y_shared[shared_index_Y] = d_y[offset]; 

        //if its first row, copy out one row above from d_x
        if (threadIdx.y == 0 && (y - 1 >= 0))
            x_shared[threadIdx.x] = d_x[(y - 1) * col_size + x]; 
        //if its last row, copy out one row below from d_x
        if ((threadIdx.y == blockDim.y - 1) && (y + 1 < row_size))
            x_shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x] = d_x[(y + 1) * col_size + x];

        if (threadIdx.x == 0 && (x - 2 >= 0))
            y_shared[threadIdx.y * (blockDim.x + 2)] = d_y[offset - 2]; 
        
        if (threadIdx.x == 0 && (x - 1 >= 0))
            y_shared[threadIdx.y * (blockDim.x + 2) + 1] = d_y[offset - 1]; 
        
    }
    
     __syncthreads(); 

    if (x < col_size && y < row_size ){
        float output = x_shared[shared_index_X] - y_shared[shared_index_Y]; 
        //coalesced access. 
        if (x - 1 >= 0) 
            output -= y_shared[shared_index_Y - 1];
        if (x - 2 >= 0)
            output -= y_shared[shared_index_Y - 2]; 

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
        printf("Program takes 2 args, <row_size> <col_size>"); 
        return 0; 
    }

    int row_size = atoi(argv[1]);
    int col_size = atoi(argv[2]); 

    float *h_x, *h_y, *h_z, *d_x, *d_y, *d_z; 
    int n_size = row_size * col_size;

    cudaHostAlloc((void**) &h_x, n_size * sizeof(float), cudaHostAllocWriteCombined); 
    cudaHostAlloc((void**) &h_y, n_size * sizeof(float), cudaHostAllocWriteCombined); 
    cudaHostAlloc((void**) &h_z, n_size * sizeof(float), cudaHostAllocWriteCombined); 
    // cudaHostAlloc((void**) &h_x, n_size * sizeof(float), 0); 
    // cudaHostAlloc((void**) &h_y, n_size * sizeof(float), 0); 
    // cudaHostAlloc((void**) &h_z, n_size * sizeof(float), 0); 
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
    int gridY = (row_size / 2 + blockSize.y - 1) / blockSize.y + 2; 
    if (gridY <= 0)
        gridY = 1; 
    dim3 gridSize(gridX, gridY); 
    

    cudaStream_t stream[N_STREAMS + 1]; 

    size_t trunkSize; 
    if (row_size % 2 == 0 || row_size == 1)
        trunkSize = col_size * (row_size / 2 + 1) * sizeof(float);
    else 
        trunkSize = col_size * (row_size / 2 + 2) * sizeof(float);



    int secondTrunkOffset = col_size * (row_size / 2 - 1); 
    if (secondTrunkOffset < 0)
        secondTrunkOffset = 0; 
    int reminderSize = n_size - n_size / 2; 

    //printf("second Trunk offset %d, trunk size %d \n", secondTrunkOffset, trunkSize);
    int rowLimits = row_size / 2 + 1;
    if (row_size % 2 == 1)
        rowLimits = row_size / 2 + 2;

    double startTime, CPUTransferFinishTime, kernelFinishTime, endTime;

    startTime = getTimeStamp(); 

    
    cudaStreamCreate(&stream[1]); 
    cudaMemcpyAsync(&d_x[0], &h_x[0], trunkSize, cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(&d_y[0], &h_y[0], trunkSize, cudaMemcpyHostToDevice, stream[1]);
    f_siggen<<<gridSize,blockSize>>>(&d_x[0], &d_y[0], &d_z[0], rowLimits, col_size); 
    cudaMemcpyAsync(&h_z[0], &d_z[0], (n_size / 2) * sizeof(float), cudaMemcpyDeviceToHost, stream[1]);


    cudaStreamCreate(&stream[2]); 
    cudaMemcpyAsync(&d_x[secondTrunkOffset], &h_x[secondTrunkOffset], trunkSize, cudaMemcpyHostToDevice, stream[2]);
    cudaMemcpyAsync(&d_y[secondTrunkOffset], &h_y[secondTrunkOffset], trunkSize, cudaMemcpyHostToDevice, stream[2]);
    cudaStreamAddCallback(stream[2], recordTimeCallback, (void*) &CPUTransferFinishTime, 0);
    f_siggen<<<gridSize,blockSize>>>(&d_x[secondTrunkOffset], &d_y[secondTrunkOffset], &d_z[secondTrunkOffset], rowLimits, col_size); 
    cudaStreamAddCallback(stream[2], recordTimeCallback, (void*) &kernelFinishTime, 0);
    cudaMemcpyAsync(&h_z[n_size / 2], &d_z[n_size / 2], reminderSize * sizeof(float), cudaMemcpyDeviceToHost, stream[2]);
   
    
    cudaError_t stream1Error = cudaStreamSynchronize(stream[1]); 
    cudaError_t stream2Error = cudaStreamSynchronize(stream[2]); 
    


    endTime = getTimeStamp(); 


    if (stream1Error != 0 || stream2Error != 0){
        printf("Error: stream 1 error %d, stream 2 error %d \n", stream1Error, stream2Error);
        cudaFree(d_x);
        cudaFree(d_z); 
        cudaFreeHost(h_x);
        cudaFreeHost(h_z);
        cudaFree(d_y);
        cudaFreeHost(h_y);
        return 2;
    }


   
    

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

            if (h_z[offset] != out){
                printf("Error: calculated difference at row %d, col %d, CPU is %f, GPU is %f \n", i, j, out, h_z[offset]);
                cudaFree(d_x);
                cudaFree(d_z); 
                cudaFreeHost(h_x);
                cudaFreeHost(h_z);
                cudaFree(d_y);
                cudaFreeHost(h_y);
                return 1;
            }
                
        }
    }

    volatile float res = h_z[5*col_size + 5]; 

    printf("%.6f %.6f %.6f %.6f %.6f \n", endTime - startTime, CPUTransferFinishTime - startTime , 
    kernelFinishTime - CPUTransferFinishTime, endTime - kernelFinishTime, res);


    cudaFree(d_x);
    cudaFree(d_z); 
    cudaFreeHost(h_x);
    cudaFreeHost(h_z);

    cudaFree(d_y);
    cudaFreeHost(h_y); 
  

    return 0;
}
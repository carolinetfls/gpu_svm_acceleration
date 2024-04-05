#include <cuda.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <random>

#define BLOCK_SIZE 32
#define TILE_WIDTH 32

// Function to compare two matrices
int compareMatrices(int* a, int* b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (a[i*cols+j] != b[i*cols+j]) {
                return 0; // Matrices are not the same
            }
        }
    }
    return 1; // Matrices are the same
}

// // Function to initialize a matrix with random values
// void initializeMatrix(int *matrix, int rows, int cols) {
//     #pragma omp parallel for
//     for (int i = 0; i < rows*cols-1; i++) {
//         matrix[i] = rand() % 1024;
//     }
// }

void initializeMatrix(int *matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(0, 1023);

    #pragma omp parallel
    {
        // Each thread gets a separate copy of the generator and distribution
        std::mt19937 gen_local(gen());
        std::uniform_int_distribution<> dis_local(dis);

        #pragma omp for
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = dis_local(gen_local);
        }
    }
}

// Function to print a matrix
void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i*cols+j]);
        }
        printf("\n");
    }
}

// Parallel matrix multiplication using OpenMP
void matrixMultiplyParallel(int *a, int *b, int *c, int m, int n, int p) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < p; j++) {
                c[i*m+j] += a[i*m+k] * b[k*n+j];
            }
        }
    }
}

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int p)
{ 
    int col = blockIdx.y * blockDim.y + threadIdx.y; 
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < p && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * p + col];
        }
        c[row * p + col] = sum;
    }
} 

//
__global__ void gpu_matrix_mult_shared_memory(int *d_a, int *d_b, int *d_result, int n) {
    __shared__ int tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tile_b[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * TILE_WIDTH + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by TILE_WIDTH
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * TILE_WIDTH + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}


float matrixMultiplyCudaBaseline(int *a, int *b, int *c, int m, int n, int p) {
    float cuda_time_used;
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

    int *a_dev, *b_dev, *c_dev;
    cudaMalloc((void **)&a_dev, m * n * sizeof(int));
    cudaMalloc((void **)&b_dev, n * p * sizeof(int));
    cudaMalloc((void **)&c_dev, m * p * sizeof(int));

    cudaMemcpy(a_dev, a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, n * p * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (p + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(start_cuda);
    // only need to modify this line
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(a_dev, b_dev, c_dev, m, n, p);
    cudaEventRecord(stop_cuda);

    cudaMemcpy(c, c_dev, m * p * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_cuda);
    cudaEventElapsedTime(&cuda_time_used, start_cuda, stop_cuda);

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    cudaEventDestroy(start_cuda);
    cudaEventDestroy(stop_cuda);

    return cuda_time_used / 1000; // Convert milliseconds to seconds
}

float matrixMultiplyCudaSharedMemory(int *a, int *b, int *c, int m, int n, int p) {
    float cuda_time_used;
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

    int *a_dev, *b_dev, *c_dev;
    cudaMalloc((void **)&a_dev, m * n * sizeof(int));
    cudaMalloc((void **)&b_dev, n * p * sizeof(int));
    cudaMalloc((void **)&c_dev, m * p * sizeof(int));

    cudaMemcpy(a_dev, a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, n * p * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (p + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(start_cuda);
    // only need to modify this line
    // gpu_matrix_mult_shared_memory<<<dimGrid, dimBlock>>>(a_dev, b_dev, c_dev, m, n, p);
    gpu_matrix_mult_shared_memory<<<dimGrid, dimBlock>>>(a_dev, b_dev, c_dev, m);
    cudaEventRecord(stop_cuda);

    cudaMemcpy(c, c_dev, m * p * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_cuda);
    cudaEventElapsedTime(&cuda_time_used, start_cuda, stop_cuda);

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    cudaEventDestroy(start_cuda);
    cudaEventDestroy(stop_cuda);

    return cuda_time_used / 1000; // Convert milliseconds to seconds
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <m> <n> <p>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);

    int *a, *b, *c_cuda, *c_cuda1;
    cudaMallocHost((void **) &a, sizeof(int)*m*n);
    cudaMallocHost((void **) &b, sizeof(int)*n*p);
    cudaMallocHost((void **) &c_cuda, sizeof(int)*m*p);
    cudaMallocHost((void **) &c_cuda1, sizeof(int)*m*p);

    initializeMatrix(a, m, n);
    initializeMatrix(b, n, p);


    // float cuda_time_baseline = matrixMultiplyCudaBaseline(a,b,c_cuda, m, n, p); 
    // printf("Time taken by CUDA baseline version: %f seconds\n", cuda_time_baseline);
    float cuda_time_shared_memory = matrixMultiplyCudaSharedMemory(a,b,c_cuda1, m, n, p); 
    printf("Time taken by CUDA shared memory version: %f seconds\n", cuda_time_shared_memory);
    // printf("The speedup is: %f times\n", cuda_time_baseline/cuda_time_shared_memory);

    float flops = 2.0f * m * n * p; // Total floating point operations
   
    // Calculate TFLOPS (TeraFLOPS)
    // float tflops_baseline = (flops / cuda_time_baseline) / 1e12f; // Convert to TeraFLOPS
    float tflops_shared_memory = (flops / cuda_time_shared_memory) / 1e12f; // Convert to TeraFLOPS

    // printf("TFLOPS for CUDA baseline version: %f TFLOPS\n", tflops_baseline);
    printf("TFLOPS for CUDA shared memory version: %f TFLOPS\n", tflops_shared_memory);
    // printf("The speedup is: %f times\n", cuda_time_baseline / cuda_time_shared_memory);

    // // Compare the results
    // if (compareMatrices(c_cuda1, c_cuda, m, p)) {
    //     printf("The results are the same.\n");
    // } else {
    //     printf("The results are different.\n");
    // }

    // printMatrix(a,m,n);
    // printMatrix(b,n,p);
    // printMatrix(c_cuda,m,p);
    // printMatrix(c_cuda1,m,p);


    // Free allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cuda); 
    cudaFree(c_cuda1); 

    return 0;
}
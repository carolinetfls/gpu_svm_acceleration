#include <cuda.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

#define BLOCK_SIZE 32


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

// Function to initialize a matrix with random values
void initializeMatrix(int *matrix, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows*cols-1; i++) {
        matrix[i] = rand() % 1024;
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
    // #pragma omp parallel for 
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

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <m> <n> <p>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);

    int *a, *b, *c_omp, *c_cuda;
    cudaMallocHost((void **) &a, sizeof(int)*m*n);
    cudaMallocHost((void **) &b, sizeof(int)*n*p);
    cudaMallocHost((void **) &c_omp, sizeof(int)*m*p);
    cudaMallocHost((void **) &c_cuda, sizeof(int)*m*p);

    initializeMatrix(a, m, n);
    initializeMatrix(b, n, p);
    // setMatrixZeros(c_omp, m, p);
    // setMatrixZeros(c_cuda, m, p);


    double start_time = omp_get_wtime();
    // Perform matrix multiplication using OpenMP
    matrixMultiplyParallel(a, b, c_omp, m, n, p);
    double end_time = omp_get_wtime();
    double omp_time = end_time - start_time;
    printf("Time taken by OpenMP version: %f seconds\n", omp_time);

    float cuda_time_used;
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda);
    // Perform matrix multiplication using CUDA
    int *a_dev, *b_dev, *c_dev;
    cudaMalloc((void **) &a_dev, m * n * sizeof(int));
    cudaMalloc((void **) &b_dev, n * p * sizeof(int));
    cudaMalloc((void **) &c_dev, m * p * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(a_dev, a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, n * p * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemset(c_dev, 0, m * p * sizeof(int));

    // Define grid and block dimensions
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (p + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Launch the kernel
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(a_dev, b_dev, c_dev, m, n, p);
    // cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(c_cuda, c_dev, m * p * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);
    cudaEventElapsedTime(&cuda_time_used, start_cuda, stop_cuda);
    printf("Time taken by CUDA version: %f milliseconds\n", cuda_time_used/1000);
    printf("Speedup: %f\n:", omp_time/(cuda_time_used/1000));
    cudaEventDestroy(start_cuda);
    cudaEventDestroy(stop_cuda);

    // printMatrix(a, m, n); 
    // printf("\n");
    // printMatrix(b, n, p);
    // printf("\n");
    // printMatrix(c_omp, m, p);
    // printf("\n");
    // printMatrix(c_cuda, m, p);

    // Compare the results
    if (compareMatrices(c_omp, c_cuda, m, p)) {
        printf("The results are the same.\n");
    } else {
        printf("The results are different.\n");
    }

    // Free allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c_omp);
    cudaFree(c_cuda); 
    // Free device memory
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    return 0;
}
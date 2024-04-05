#include "svm.h"
#define DEBUG_SVMTOCONJ_SEQ_TIME 1
#define DEBUG_CONJU_SEQ_TIME 1
extern void computeRBFKernelCUDA(const double *trainData, double *kernelMatrix, int featureLen, double sigma, int trainSampleCnt);

// Utility function to convert single precision data to double precision
double *convertToDouble(float **data, int sampleCnt, int featureLen)
{
        double *d_data = (double *)malloc(sampleCnt * featureLen * sizeof(double));
        for (int i = 0; i < sampleCnt; ++i)
        {
                for (int j = 0; j < featureLen; ++j)
                {
                        d_data[i * featureLen + j] = (double)data[i][j];
                }
        }
        return d_data;
}

// Utility function to convert double precision data to single precision
void convertToFloat(double *d_data, float **data, int sampleCnt)
{
        for (int i = 0; i < sampleCnt; ++i)
        {
                for (int j = 0; j < sampleCnt; ++j)
                {
                        data[i][j] = (float)d_data[i * sampleCnt + j];
                }
        }
}

__global__ void polynomialKernelCUDA(const double *d_trainData, double *d_kernelMatrix, int featureLen, int mDegree, double c, int trainSampleCnt)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i < trainSampleCnt && j < trainSampleCnt)
        {
                double innerProd = 0.0;
                for (int k = 0; k < featureLen; k++)
                {
                        innerProd += d_trainData[i * featureLen + k] * d_trainData[j * featureLen + k];
                }
                d_kernelMatrix[i * trainSampleCnt + j] = pow(c + innerProd, mDegree);
        }
}

void computePolynomialKernelCUDA(const double *trainData, double *kernelMatrix, int featureLen, int mDegree, double c, int trainSampleCnt)
{
        double *d_trainData, *d_kernelMatrix;

        cudaMalloc((void **)&d_trainData, trainSampleCnt * featureLen * sizeof(double));
        cudaMalloc((void **)&d_kernelMatrix, trainSampleCnt * trainSampleCnt * sizeof(double));

        cudaMemcpy(d_trainData, trainData, trainSampleCnt * featureLen * sizeof(double), cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((trainSampleCnt + blockSize.x - 1) / blockSize.x, (trainSampleCnt + blockSize.y - 1) / blockSize.y);

        polynomialKernelCUDA<<<gridSize, blockSize>>>(d_trainData, d_kernelMatrix, featureLen, mDegree, c, trainSampleCnt);

        cudaMemcpy(kernelMatrix, d_kernelMatrix, trainSampleCnt * trainSampleCnt * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_trainData);
        cudaFree(d_kernelMatrix);
}

int main(int argc, char *argv[])
{
        char *kernel = NULL, *trainPath = NULL, *valPath = NULL;
        int featureLen = 0, trainSampleCnt = 0, valSampleCnt = 0;

        int C = -1; // limit the support vector number, the upper limit of alpha
        float learnRate = -1.0;
        float limit = -1.0; // gradient limit
        int i, j;
        int mDegree = 0;  // parameter for Polynomial Kernel
        float c = 1.0;    // parameter for Quadratic kernel
        float sigma = -1; // parameter for Radial Basis Function (RBF) kernel
        int epochNum = 10000;
        int dimBlock = 8;
        Data_p trainData, valData, trainDataOrigin, valDataOrigin;

        // 1. set up phase
        // analyze input, get kernel type, train path and test path
        const char *optstring = "k:t:v:l:c:n:e:C:r:m:s:p:d:";
        int opt;
        while ((opt = getopt(argc, argv, optstring)) != -1)
        {
                switch (opt)
                {
                case 'k':
                        kernel = optarg;
                        break;
                case 't':
                        trainPath = optarg;
                        break;
                case 'v':
                        valPath = optarg;
                        break;
                case 'l':
                        featureLen = atoi(optarg);
                        break;
                case 'c':
                        trainSampleCnt = atoi(optarg);
                        break;
                case 'n':
                        valSampleCnt = atoi(optarg);
                        break;
                case 'm':
                        mDegree = atoi(optarg);
                        break;
                case 's':
                        sigma = atof(optarg);
                        break;
                case 'e':
                        limit = atof(optarg);
                        break;
                case 'C':
                        C = atoi(optarg);
                        break;
                case 'r':
                        learnRate = atof(optarg);
                        break;
                case 'p':
                        epochNum = atoi(optarg);
                        break;
                case 'd':
                        dimBlock = atoi(optarg);
                        break;
                case '?':
                        printf("Usage: ./kernel_svm [-k kernel] [-t train_data_path] [-v valid_data_path] [-l feature_len] [-c trainSampleCnt] [-n valSampleCnt] [-e limit] [-C C] [-r learningRate] [-m polynomial_kernel_m_degree] [-s RBF_kernel_sigma] [-p epochNum] [-d dimBlock]\n");
                        return -1;
                }
        }
        if (kernel == NULL || trainPath == NULL || valPath == NULL || featureLen == 0 || trainSampleCnt == 0 || valSampleCnt == 0 || limit < 0 || C < 0 || learnRate < 0)
        {
                printf("input not valid!\n");
                return -1;
        }

        if (kernel[0] == 'p' && mDegree == 0)
        {
                printf("the m degree of polynomial kernel is not valid!\n");
                return -1;
        }

        if (kernel[0] == 'r' && sigma < 0)
        {
                printf("the sigma of RBF kernel is not valid!\n");
                return -1;
        }

        // 2. get data from paths
        getData(&trainDataOrigin, trainPath, featureLen, trainSampleCnt);
        getData(&valDataOrigin, valPath, featureLen, valSampleCnt);

        trainData.feature = (float **)malloc(trainSampleCnt * sizeof(float *));
        for (i = 0; i < trainSampleCnt; i++)
                trainData.feature[i] = (float *)malloc(featureLen * sizeof(float));
        trainData.target = trainDataOrigin.target;

        valData.feature = (float **)malloc(valSampleCnt * sizeof(float *));
        for (i = 0; i < valSampleCnt; i++)
                valData.feature[i] = (float *)malloc(featureLen * sizeof(float));
        valData.target = valDataOrigin.target;

        // 3.perforem feature normalization
        featureNorm((const float **)trainDataOrigin.feature, (const float **)valDataOrigin.feature, trainData.feature, valData.feature, featureLen, trainSampleCnt, valSampleCnt);

        // 4. construct kernel matrix
        float (*kernelFunction)(const float *, const float *, int, int, float, float) = NULL;
        switch (kernel[0])
        {
        case 'l':
                kernelFunction = linearKernel;
                break;
        case 'q':
                kernelFunction = quadraticKernel;
                break;
        case 'p':
                kernelFunction = polynomialKernel;
                break;
        case 'r':
                kernelFunction = rbfKernel;
                break;
        default:
                kernelFunction = linearKernel;
                break; // Handle default kernel type
        }

        float **kernelMatrix = (float **)malloc(trainSampleCnt * sizeof(float *));
        if (kernelMatrix == NULL)
        {
                perror("Failed to allocate memory for kernelMatrix");
                exit(EXIT_FAILURE);
        }

        for (i = 0; i < trainSampleCnt; i++)
        {
                kernelMatrix[i] = (float *)malloc(trainSampleCnt * sizeof(float));
                if (kernelMatrix[i] == NULL)
                {
                        perror("Failed to allocate memory for kernelMatrix[i]");
                        for (j = 0; j < i; j++)
                                free(kernelMatrix[j]);
                        free(kernelMatrix);
                        exit(EXIT_FAILURE);
                }
        }

        if (kernel[0] == 'r')
        {
                // Convert trainData.feature to double precision for CUDA computation
                double *d_trainData = convertToDouble(trainData.feature, trainSampleCnt, featureLen);
                double *d_kernelMatrix = (double *)malloc(trainSampleCnt * trainSampleCnt * sizeof(double));

                // Start measuring time
                clock_t start_time = clock();

                // Compute the RBF kernel matrix using CUDA
                computeRBFKernelCUDA(d_trainData, d_kernelMatrix, featureLen, sigma, trainSampleCnt);

                // Stop measuring time
                clock_t end_time = clock();

                // Calculate the elapsed time in seconds
                double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
                printf("Time taken for RBF kernel computation: %f seconds\n", elapsed_time);

                // Convert the result back to single precision
                convertToFloat(d_kernelMatrix, kernelMatrix, trainSampleCnt);

                // Free the double precision train data and kernel matrix
                free(d_trainData);
                free(d_kernelMatrix);
        }
        else if (kernel[0] == 'p')
        {
                // Convert trainData.feature to double precision for CUDA computation
                double *d_trainData = convertToDouble(trainData.feature, trainSampleCnt, featureLen);
                double *d_kernelMatrix = (double *)malloc(trainSampleCnt * trainSampleCnt * sizeof(double));

                // Start measuring time
                clock_t start_time = clock();

                // Compute the Polynomial kernel matrix using CUDA
                computePolynomialKernelCUDA(d_trainData, d_kernelMatrix, featureLen, mDegree, c, trainSampleCnt);

                // Stop measuring time
                clock_t end_time = clock();

                // Calculate the elapsed time in seconds
                double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
                printf("Time taken for Polynomial kernel computation: %f seconds\n", elapsed_time);

                // Convert the result back to single precision
                convertToFloat(d_kernelMatrix, kernelMatrix, trainSampleCnt);

                // Free the double precision train data and kernel matrix
                free(d_trainData);
                free(d_kernelMatrix);
        }
        else
        {
                for (i = 0; i < trainSampleCnt; i++)
                {
                        for (j = 0; j <= i; j++)
                        {
                                kernelMatrix[i][j] = kernelFunction((const float *)trainData.feature[i], (const float *)trainData.feature[j], featureLen, mDegree, c, sigma);
                                kernelMatrix[j][i] = kernelMatrix[i][j];
                        }
                }
        }

        // 5. training
        // 5.1 transfer from svm to conjugate form
        int optMatrixLen = trainSampleCnt - 1;
        float *A = (float *)calloc(optMatrixLen * optMatrixLen, sizeof(float));
        if (!A)
        {
                fprintf(stderr, " Cannot allocate the %u x %u vector A\n", optMatrixLen, optMatrixLen);
                exit(1);
        }
        float bias = 0;
        float *b = (float *)calloc(optMatrixLen, sizeof(float));

        for (i = 0; i < optMatrixLen; i++)
        {
                b[i] = trainData.target[i] - trainData.target[optMatrixLen];
        }

        clock_t start, end;
        float time_taken;
#ifdef DEBUG_SVMTOCONJ_SEQ_TIME
        float *A_seq = (float *)calloc(optMatrixLen * optMatrixLen, sizeof(float));
        start = clock();
        svmToConjugateCPU((const int *)trainData.target, (const float **)kernelMatrix, A_seq, optMatrixLen, C);

        end = clock();
        time_taken = ((float)(end - start)) / CLOCKS_PER_SEC;
        printf("svmToConju CPU Time taken = %lf\n", time_taken);
#endif

        start = clock();
        svmToConjugateGPU((const int *)trainData.target, (const float **)kernelMatrix, A, optMatrixLen, C, dimBlock);
        end = clock();
        time_taken = ((float)(end - start)) / CLOCKS_PER_SEC;
        printf("svmToConju GPU Time taken = %lf\n", time_taken);

        int isSym = isSymMatrix((const float *)A, optMatrixLen);

        // 5.2 conjugate descent training
        float *alpha = (float *)calloc(trainSampleCnt, sizeof(float));
        float eps = 0.001;

#ifdef DEBUG_CONJU_SEQ_TIME
        float *alpha_seq = (float *)calloc(trainSampleCnt, sizeof(float));
        start = clock();
        conjugateDescCPU((const float *)A, (const float *)b, alpha_seq, epochNum, eps, optMatrixLen);
        end = clock();
        time_taken = ((float)(end - start)) / CLOCKS_PER_SEC;
        printf("conjugate CPU Time taken = %lf\n", time_taken);
#endif

        start = clock();
        conjugateDescGPU((const float *)A, (const float *)b, alpha, epochNum, eps, optMatrixLen, dimBlock);
        end = clock();
        time_taken = ((float)(end - start)) / CLOCKS_PER_SEC;
        printf("conjugate GPU Time taken = %lf\n", time_taken);

        for (i = 0; i < optMatrixLen; i++)
                alpha[optMatrixLen] += (-1) * alpha[i];

        bias = getBias((const int *)trainData.target, (const float **)kernelMatrix, (const float *)alpha, optMatrixLen, C);

        // 6.check val result
        float train_acc = getTrainAccuracy((const float *)alpha, (const int *)trainData.target, (const float **)kernelMatrix, trainSampleCnt, bias);
        float val_acc = getValAccuracy((const float *)alpha, (const float **)trainData.feature, (const float **)valData.feature, (const int *)valData.target, (const char *)kernel, trainSampleCnt, valSampleCnt, bias, featureLen, mDegree, c, sigma);

        // 7. free the memory
        for (i = 0; i < trainSampleCnt; i++)
                free(kernelMatrix[i]);
        free(kernelMatrix);

        for (i = 0; i < trainSampleCnt; i++)
        {
                free(trainData.feature[i]);
                free(trainDataOrigin.feature[i]);
        }
        free(trainData.feature);
        free(trainData.target);
        free(trainDataOrigin.feature);

        for (i = 0; i < valSampleCnt; i++)
        {
                free(valData.feature[i]);
                free(valDataOrigin.feature[i]);
        }
        free(valData.feature);
        free(valData.target);
        free(valDataOrigin.feature);

        free(A);
        free(alpha);
        free(b);
        return 0;
}

void getData(Data_p *data, char *path, int featureLen, int sampleCnt)
{
        /*
          get the data from its path
          the data format:
          each line is a sample
          in each line: feature 1, feature 2, ..., feature n, target
        */
        FILE *file = fopen(path, "r");
        if (file == NULL)
        {
                perror("Error while opening the file.\n");
                exit(EXIT_FAILURE);
        }

        data->feature = (float **)malloc(sampleCnt * sizeof(float *));
        data->target = (int *)malloc(sampleCnt * sizeof(int));
        int i = 0;
        int j = 0;
        char line[MAX_LINE_LENGTH];
        while (fgets(line, MAX_LINE_LENGTH, file) && i < sampleCnt)
        {
                j = 0;
                char *token = strtok(line, ",");
                data->feature[i] = (float *)malloc(featureLen * sizeof(float));
                while (token != NULL)
                {
                        if (j < featureLen)
                                data->feature[i][j++] = atof(token);
                        else
                                data->target[i] = atoi(token);
                        token = strtok(NULL, ",");
                }
                i++;
        }
        fclose(file);
}

float getTrainAccuracy(const float *alpha, const int *dataTarget, const float **kernelMatrix, int sampleCnt, float bias)
{
        /*
        get train data accuracy
        */
        int rightCnt = 0;
        int i, j;
        for (i = 0; i < sampleCnt; i++)
        {
                float res = 0.0;
                for (j = 0; j < sampleCnt; j++)
                {
                        res += alpha[j] * kernelMatrix[j][i];
                }
                res += bias;
                if (res * dataTarget[i] > 0)
                        rightCnt += 1;
        }
        return rightCnt / sampleCnt;
}

float getValAccuracy(const float *alpha, const float **trainDataFeature, const float **valDataFeature, const int *valDataTarget, const char *kernel, int trainSampleCnt, int valSampleCnt, float bias, int featureLen, int mDegree, float c, float sigma)
{
        /*
        get validation data accuracy
        */
        int rightCnt = 0;
        int i, j;
        float (*kernelFunction)(const float *, const float *, int, int, float, float) = NULL;
        switch (kernel[0])
        {
        case 'l':
                kernelFunction = linearKernel;
                break;
        case 'q':
                kernelFunction = quadraticKernel;
                break;
        case 'p':
                kernelFunction = polynomialKernel;
                break;
        case 'r':
                kernelFunction = rbfKernel;
                break;
        default:
                kernelFunction = linearKernel;
                break; // Handle default kernel type
        }
        for (i = 0; i < valSampleCnt; i++)
        {
                float res = 0.0;
                for (j = 0; j < trainSampleCnt; j++)
                {
                        res += alpha[j] * kernelFunction(trainDataFeature[j], valDataFeature[i], featureLen, mDegree, c, sigma);
                }
                res += bias;
                if (res * valDataTarget[i] > 0)
                        rightCnt += 1;
        }
        return rightCnt / valSampleCnt;
}

int isSymMatrix(const float *A, int size)
{

        // check if a matrix is symmetric

        int isSym = 1;
        int i, j;
        for (i = 0; i < size; i++)
        {
                for (j = i + 1; j < size; j++)
                {
                        if (!isSame(A[i * size + j], A[j * size + i]))
                                isSym = 0;
                }
        }
        return isSym;
}

int isSame(float a, float b)
{
        /*
        check if two float numbers are close enough
        */
        return fabs(a - b) < 1e-6;
}

int matrixMul(const float **trainDataFeature, float **kernelMatrix, int trainSampleCnt)
{
        /*
         trainDataFeature matrix and its transpose's inner product
         as trainDataFeature matrix is a symmetric matrix,
         for each result element, trainDataFeature[i][k] * trainDataFeature[j][k]
         is equivalent to trainDataFeature[i][k] * trainDataFeature[k][j]
        */
        int i, j, k;
        for (i = 0; i < trainSampleCnt; i++)
        {
                for (j = 0; j < trainSampleCnt; j++)
                {
                        kernelMatrix[i][j] = 0;
                        for (k = 0; k < trainSampleCnt; k++)
                        {
                                kernelMatrix[i][j] += trainDataFeature[i][k] * trainDataFeature[j][k];
                        }
                }
        }
        return 0;
}

int featureNorm(const float **trainDataFeature, const float **valDataFeature, float **trainDataNormFeature, float **valDataNormFeature, int featureLen, int trainSampleCnt, int valSampleCnt)
{
        /*
         perform min-max normalization to make input feature within 0-1
        */
        int i, j;
        float minArr[featureLen];
        float maxArr[featureLen];
        for (i = 0; i < featureLen; i++)
        {
                minArr[i] = INT_MAX;
                maxArr[i] = INT_MIN;
        }
        for (i = 0; i < featureLen; i++)
                for (j = 0; j < trainSampleCnt; j++)
                {
                        if (trainDataFeature[j][i] < minArr[i])
                                minArr[i] = trainDataFeature[j][i];
                        if (trainDataFeature[j][i] > maxArr[i])
                                maxArr[i] = trainDataFeature[j][i];
                }

        for (i = 0; i < featureLen; i++)
        {
                float dist = maxArr[i] - minArr[i];
                for (j = 0; j < trainSampleCnt; j++)
                        trainDataNormFeature[j][i] = (trainDataFeature[j][i] - minArr[i]) / dist;
                for (j = 0; j < valSampleCnt; j++)
                        valDataNormFeature[j][i] = (valDataFeature[j][i] - minArr[i]) / dist;
        }

        return 0;
}

__device__ double atomicAddDouble(double *address, double val)
{
        unsigned long long int *address_as_ull = (unsigned long long int *)address;
        unsigned long long int old = *address_as_ull, assumed;

        do
        {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                                __double_as_longlong(val + __longlong_as_double(assumed)));

                // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
}

__global__ void linearKernelKernel(const double *x1, const double *x2, double *result, int featureLen)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < featureLen)
        {
                atomicAddDouble(result, x1[tid] * x2[tid]);
        }
}

double gpu_linearKernel(const double *x1, const double *x2, int featureLen, int mDegree, double c, double sigma)
{
        double *d_result;
        double h_result = 0.0;

        // Allocate memory on the device for the result
        cudaMalloc((void **)&d_result, sizeof(double));
        cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);

        // Define block and grid dimensions
        int blockSize = 256;
        int gridSize = (featureLen + blockSize - 1) / blockSize;

        // Launch the CUDA kernel
        linearKernelKernel<<<gridSize, blockSize>>>(x1, x2, d_result, featureLen);

        // Copy the result back to the host
        cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        // Free allocated memory on the device
        cudaFree(d_result);

        // Return the result
        return h_result;
}

// CUDA kernel function to compute the RBF kernel matrix
__global__ void rbfKernelCUDA(const double *d_trainData, double *d_kernelMatrix, int featureLen, double sigma, int trainSampleCnt)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i < trainSampleCnt && j < trainSampleCnt)
        {
                double squaredDistance = 0.0;
                for (int k = 0; k < featureLen; k++)
                {
                        double diff = d_trainData[i * featureLen + k] - d_trainData[j * featureLen + k];
                        squaredDistance += diff * diff;
                }
                d_kernelMatrix[i * trainSampleCnt + j] = exp(-squaredDistance / (2 * sigma * sigma));
        }
}

// Function to compute the RBF kernel matrix using CUDA
void computeRBFKernelCUDA(const double *trainData, double *kernelMatrix, int featureLen, double sigma, int trainSampleCnt)
{
        double *d_trainData, *d_kernelMatrix;

        cudaMalloc((void **)&d_trainData, trainSampleCnt * featureLen * sizeof(double));
        cudaMalloc((void **)&d_kernelMatrix, trainSampleCnt * trainSampleCnt * sizeof(double));

        cudaMemcpy(d_trainData, trainData, trainSampleCnt * featureLen * sizeof(double), cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((trainSampleCnt + blockSize.x - 1) / blockSize.x, (trainSampleCnt + blockSize.y - 1) / blockSize.y);

        rbfKernelCUDA<<<gridSize, blockSize>>>(d_trainData, d_kernelMatrix, featureLen, sigma, trainSampleCnt);

        cudaMemcpy(kernelMatrix, d_kernelMatrix, trainSampleCnt * trainSampleCnt * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_trainData);
        cudaFree(d_kernelMatrix);
}

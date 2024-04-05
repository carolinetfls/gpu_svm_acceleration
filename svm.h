
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <limits.h>
//it defines each line's maximum length, should be set to a larger number if input dimension is big
#define MAX_LINE_LENGTH 1024
//#define DEBUG_SVMTOCONJ_SEQ_TIME 1
//#define DEBUG_CONJU_SEQ_TIME 1

typedef struct{
     float** feature;
     int* target;
}Data_p; // for represent trainData and valData
//feature -> sampleCnt * featureLen
//target ->  sampleCnt * 1

//global variables
//Data_p trainData, valData, trainDataOrigin, valDataOrigin;

//functions
float generateGuassRandom();
void getData(Data_p* data, char* Path, int featureLen, int sampleCnt);
int featureNorm(const float** trainDataFeature, const float** valDataFeature, float** trainDataNormFeature, float** valDataNormFeature, int featureLen, int trainSampleCnt, int valSampleCnt);
float linearKernel(const float* x1, const float* x2, int featureLen, int mDegree, float c, float sigma);
float quadraticKernel(const float* x1, const float* x2, int featureLen, int mDegree, float c, float sigma);
float polynomialKernel(const float* x1, const float* x2, int featureLen, int mDegree, float c, float sigma);
float rbfKernel(const float* x1, const float* x2, int featureLen, int mDegree, float c, float sigma);
int matrixMul(const float ** trainDataFeature, float** kernelMatrix, int trainSampleCnt);
float getTrainAccuracy(const float* alpha, const int* dataTarget, const float** kernelMatrix, int sampleCnt, float bias);
float getValAccuracy(const float* alpha, const float** trainDataFeature, const float** valDataFeature, const int* valDataTarget, const char* kernel, int trainSampleCnt, int valSampleCnt, float bias, int featureLen, int mDegree, float c, float sigma);
int svmToConjugateCPU(const int* trainDataTarget, const float** kernelMatrix, float* A, int optMatrixLen, int C);
int svmToConjugateGPU(const int* trainDataTarget, const float** kernelMatrix, float* A, int optMatrixLen, int C, int dimBlock);
float getBias(const int* trainDataTarget, const float** kernelMatrix, const float* alpha, int optMatrixLen, int C);
int isSymMatrix(const float* A, int size);
int isSame(float a, float b);
__global__  void SvmToConjugateDev(float* kernel_mat_d, float* q_d, float* A_d, int n, int C);
__global__ void ConjugateDescDevForAp(float* A, float* p, float* ap, int n);
__global__ void ConjugateDescDevForAlpha(float* ap, float* p, float* alpha, int n);
__global__ void ConjugateDescDevForAlphaFinal(float* ap, float* rrPrev);
__global__ void ConjugateDescDevForXAndR(float* ap, float* p, float* alpha, float* r, float*   x, int n);
__global__ void ConjugateDescDevForBeta(float* r, float * rrCur, int n);
 __global__ void ConjugateDescDevForBetaFinal(float* rrCur, float * rrPrev, float* beta);
__global__ void ConjugateDescDevForP(float* r, float * beta, float* p, int n) ;
int conjugateDescCPU(const float* A, const float* b, float* x, int max_ite, float eps, int n_sample);
int conjugateDescGPU(const float* A, const float* b, float* x, int maxIte, float eps, int nSample, int dimBlock);

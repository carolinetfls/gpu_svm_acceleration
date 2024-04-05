#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
//#include <cuda.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <limits.h>
//#define DEBUG_CONJU 1
//#define DEBUG_SVMTOCONJU 1
//#define DEBUG_GEN_DATA 1
//#define DEBUG_GET_DATA 1
//it defines each line's maximum length, should be set to a larger number if input dimension is big
#define MAX_LINE_LENGTH 1024

//sequential code for svm

typedef struct{
     double** feature;
     int* target;
}Data_p; // for represent trainData and valData
//feature -> sampleCnt * featureLen
//target ->  sampleCnt * 1

//global variables
Data_p trainData, valData, trainDataNorm, valDataNorm;

//functions
double generateGuassRandom();
void getData(Data_p* data, char* Path, int featureLen, int sampleCnt);
int featureNorm(const double** trainDataFeature, const double** valDataFeature, double** trainDataNormFeature, double** valDataNormFeature, int featureLen, int trainSampleCnt, int valSampleCnt);
//optimization target
double linearKernel(const double* x1, const double* x2, int featureLen, int mDegree, double c, double sigma);
double quadraticKernel(const double* x1, const double* x2, int featureLen, int mDegree, double c, double sigma);
double polynomialKernel(const double* x1, const double* x2, int featureLen, int mDegree, double c, double sigma);
double rbfKernel(const double* x1, const double* x2, int featureLen, int mDegree, double c, double sigma);
//performance
double computeObj(const double* alpha, const Data_p* trainData, const double** kernelMatrix, double beta, int trainSampleCnt);
bool updateAlpha(double* alpha, const Data_p* trainData, const double** kernelMatrix, double beta, double learnRate, int trainSampleCnt, int C, double limit);
void updateBeta(double* beta, const double* alpha, const Data_p* trainData, int trainSampleCnt);
double getTrainAccuracy(const double* alpha, const int* dataTarget, const double** kernelMatrix, int sampleCnt, double bias);
double getValAccuracy(const double* alpha, const double** trainDataFeature, const double** valDataFeature, const int* valDataTarget, const char* kernel, int trainSampleCnt, int valSampleCnt, double bias, int featureLen, int mDegree, double c, double sigma);
double getTrainAccuracyWithWeight(const double* weight, double bias, const int* dataTarget, const double** dataFeature, int sampleCnt, int featureLen);
int conjugateDesc(const double** A, const double* b, double* x, int max_ite, double eps, int n_sample);
int svmToConjugate(const int* trainDataTarget, const double** kernelMatrix, double** A, double* b, int optMatrixLen, int C);
double getBias(const int* trainDataTarget, const double** kernelMatrix, const double* alpha, int optMatrixLen, int C);
int isSymMatrix(const double** A, int size);
void printInSym(const double** A, int size);
int isSame(double a, double b);
//void fromAlphaToWeight(const double* alpha, const double** trainDataFeature, double* weight, int sampleCnt, int featureLen);
int checkConjuResWithOrig(const double** kernelMatrix, const double* alpha, double bias, const int* target, int sampleCnt);
int matrixMul(const double ** trainDataFeature, double** kernelMatrix, int trainSampleCnt);

int main(int argc, char *argv[]){
    int tmp;
    char *kernel= NULL, *trainPath = NULL, *valPath = NULL;
    int featureLen = 0, trainSampleCnt = 0, valSampleCnt = 0;

    int C = -1;//limit the support vector number, the upper limit of alpha
    double learnRate = -1.0;
    double limit = -1.0;//gradient limit
    int i,j;
    int mDegree = 0;//parameter for Polynomial Kernel
    double c = 1.0; //parameter for Quadratic kernel
    double sigma = -1;//parameter for Radial Basis Function (RBF) kernel
    int epochNum = 1000; 

    //1. analyze input, get kernel type, train path and test path
    const char* optstring = "k:t:v:l:c:n:m:s:e:C:r:";
    int opt;
    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 'k': kernel = optarg; break;
            case 't': trainPath = optarg; break;
            case 'v': valPath = optarg; break;
            case 'l': featureLen = atoi(optarg); break;
            case 'c': trainSampleCnt = atoi(optarg); break;
            case 'n': valSampleCnt = atoi(optarg); break;
            case 'm': mDegree = atoi(optarg); break;
            case 's': sigma = atof(optarg); break;
            case 'e': limit = atof(optarg); break;
            case 'C': C = atoi(optarg); break;
            case 'r': learnRate = atof(optarg); break;
            case 'p': epochNum = atoi(optarg); break; 
            case '?':
                printf("Usage: ./kernel_svm [-k kernel] [-t train_data_path] [-v valid_data_path] [-l feature_len] [-c trainSampleCnt] [-n valSampleCnt] [-e limit] [-C C] [-r learningRate] [-m polynomial_kernel_m_degree] [-s RBF_kernel_sigma] [-p epochNum]\n");
                return -1;
        }
    }
    if (kernel == NULL || trainPath == NULL || valPath == NULL || featureLen == 0 || trainSampleCnt == 0 || valSampleCnt == 0 || limit < 0 || C < 0 || learnRate < 0){
        printf("input not valid!\n");
        return -1;
    }

    if (kernel[0] == 'p' && mDegree == 0){
        printf("the m degree of polynomial kernel is not valid!\n");
        return -1;
    }

    if (kernel[0] == 'r' && sigma < 0){
        printf("the sigma of RBF kernel is not valid!\n");
        return -1;
    }
   
    //2. get data from paths
    getData(&trainData, trainPath, featureLen, trainSampleCnt);
    getData(&valData, valPath, featureLen, valSampleCnt);
    //feature normalization
    /*
    trainDataNorm.feature = (double**)malloc(trainSampleCnt*sizeof(double*));
    for (i=0; i<trainSampleCnt; i++)
        trainDataNorm.feature[i] = (double*)malloc(featureLen*sizeof(double));
    trainDataNorm.target = trainData.target;

    valDataNorm.feature = (double**)malloc(valSampleCnt*sizeof(double*));
    for (i=0; i<valSampleCnt; i++)
        valDataNorm.feature[i] = (double*)malloc(featureLen*sizeof(double));
    valDataNorm.target = valData.target;
    
    featureNorm((const double**)trainData.feature, (const double**) valData.feature, trainDataNorm.feature, valDataNorm.feature, featureLen, trainSampleCnt, valSampleCnt);

    double kernel_00 = 0;
    for (i=0; i<featureLen; i++){
        printf("train sample feature %d: %f\n", i, (trainDataNorm.feature)[0][i]);
    }
    
    for (i=0; i<featureLen; i++){
        printf("val sample feature %d: %f\n", i, (valDataNorm.feature)[0][i]);
    }*/
    #ifdef DEBUG_GET_DATA
    for (i=0; i< 5; i++){
        printf("train sample %d ", i);
        for (j=0; j<featureLen; j++){
            printf("feature %d: %f ", j, (trainData.feature)[i][j]);
        }
     printf(" target %d\n", trainData.target[i]);
     }
    #endif

    //3. get kernel matrix 
    double (*kernelFunction)(const double*, const double*, int, int, double, double) = NULL;
    switch (kernel[0]) {
            case 'l': kernelFunction = linearKernel; break;
            case 'q': kernelFunction = quadraticKernel; break;
            case 'p': kernelFunction = polynomialKernel; break;
            case 'r': kernelFunction = rbfKernel; break;
            default: kernelFunction = linearKernel; break; // Handle default kernel type
    }
   
    double** kernelMatrix = (double**)malloc(trainSampleCnt*sizeof(double*));
    if (kernelMatrix == NULL) {
        perror("Failed to allocate memory for kernelMatrix");
        exit(EXIT_FAILURE);
    }

    for (i=0; i < trainSampleCnt; i++) {
        kernelMatrix[i] = (double*)malloc(trainSampleCnt*sizeof(double));
        if (kernelMatrix[i] == NULL) {
            perror("Failed to allocate memory for kernelMatrix[i]");
            for (j = 0; j < i; j++) free(kernelMatrix[j]);
            free(kernelMatrix);
            exit(EXIT_FAILURE);
        }
    }

    if (kernel[0] == 'r'){
        matrixMul((const double **)trainData.feature, kernelMatrix, trainSampleCnt);
    }
    else{
            for (i=0; i<trainSampleCnt; i++){
                    for (j=0; j <= i; j++){
                            kernelMatrix[i][j] = kernelFunction((const double*)trainData.feature[i], (const double*)trainData.feature[j], featureLen, mDegree, c, sigma);
                            kernelMatrix[j][i] = kernelMatrix[i][j];
                }
            }
    }
    
    //4. training
    //4.1 transfer from svm to conjugate form
   #ifdef DEBUG_SVMTOCONJU
   trainData.target[0] = 1;
   trainData.target[1] = 2;
   trainData.target[2] = 3;
   trainData.target[3] = 4;
   trainData.target[4] = 5;
   double line_one[5] = {4, 2, 1, 0, 0};
   double line_two[5] = {2, 4, 2, 1, 0};
   double line_three[5] = {1, 2, 4, 2, 1};
   double line_four[5] = {0, 1, 2, 4, 2};
   double line_five[5] = {0, 0, 1, 2, 4};
   
   for (i=0; i<5; i++){
        kernelMatrix[0][i] = line_one[i];
        kernelMatrix[1][i] = line_two[i];
        kernelMatrix[2][i] = line_three[i];
        kernelMatrix[3][i] = line_four[i];
        kernelMatrix[4][i] = line_five[i];
   }
   #endif
   
   int optMatrixLen = trainSampleCnt-1;
   double** A = (double**)malloc(optMatrixLen*sizeof(double*));
   double bias = 0;
   for (i=0; i<optMatrixLen; i++){
        A[i] = (double*) malloc(optMatrixLen*sizeof(double));
   }
   double *b = (double*)malloc(optMatrixLen*sizeof(double));
   int isSym = isSymMatrix((const double**)kernelMatrix, trainSampleCnt);
   printf("before convert to conjugate form is kernelMatrix symmetrical: %d\n", isSym);
   svmToConjugate((const int*)trainData.target, (const double**)kernelMatrix, A, b, optMatrixLen, C); 
   isSym = isSymMatrix((const double**)A, optMatrixLen);
   printf("after convert to conjugate form is A symmetrical: %d\n", isSym);
   #ifdef DEBUG_SVMTOCONJU
   printInSym((const double**)A, optMatrixLen);
   #endif
  
   #ifdef DEBUG_SVMTOCONJU  
   for (i=0; i<optMatrixLen; i++){
        if (i%100 ==0)
            printf("line i: ");
        for (j=0; j<optMatrixLen; j++){
            if((i%100 == 0) && (j%100 == 0)){
                printf(" %d: %lf;", j, A[i][j]);
            }
        }
        if (i%100 == 0)
            printf("\n");
   }
   for (i=0; i<optMatrixLen; i++){
        if(i%100 == 0)
            printf("b[%d]: %f\n", i, b[i]);
   }
   #endif

   //4.2 conjugate descent training  
   #ifdef DEBUG_CONJU
   double line_one[5] = {4, 2, 1, 0, 0};
   double line_two[5] = {2, 4, 2, 1, 0};
   double line_three[5] = {1, 2, 4, 2, 1};
   double line_four[5] = {0, 1, 2, 4, 2};
   double line_five[5] = {0, 0, 1, 2, 4};
   for (i=0; i<5; i++){
        A[0][i] = line_one[i];
        A[1][i] = line_two[i];
        A[2][i] = line_three[i];
        A[3][i] = line_four[i];
        A[4][i] = line_five[i];
   }
   
   double b_val[5] =  {1, 2, 3, 4, 5};
   for (i=0; i<5; i++)
       b[i] = b_val[i];
   }
   #endif
   
   double* alpha = (double*)calloc(trainSampleCnt, sizeof(double));
   double eps = 0.001;
    conjugateDesc((const double**)A, (const double*)b, alpha, 100, eps, optMatrixLen);   
    for (i=0; i<optMatrixLen; i++){
        if (i%100 == 0)
            printf("after alpha[%d] = %f\n", i, alpha[i]);
    }
   for (i=0; i<optMatrixLen; i++)
    alpha[optMatrixLen] += (-1) * alpha[i];
   //#ifdef  DEBUG_SVMTOCONJU 
   /*double* weight = (double*)calloc(featureLen, sizeof(double));
    fromAlphaToWeight((const double*) alpha, (const double**)trainData.feature, weight, trainSampleCnt, featureLen);
   printf("weight ");
   for (i=0; i<featureLen; i++)
        printf("%d : %f ", i, weight[i]);
   printf("\n");*/
    //#endif

    bias = getBias((const int*)trainData.target, (const double**)kernelMatrix, (const double*)alpha, optMatrixLen, C);
    printf("bias %f\n", bias);
    
    #ifdef DEBUG_SVMTOCONJU 
    int isSame = checkConjuResWithOrig((const double**) kernelMatrix, (const double*) alpha, bias, (const int*) trainData.target, trainSampleCnt);
    printf("isSame %d\n", isSame);
    #endif

    //5.check val result
    double train_acc = getTrainAccuracy((const double*)alpha, (const int*)trainData.target, (const double**)kernelMatrix, trainSampleCnt, bias);
    //double train_acc_weight = getTrainAccuracyWithWeight((const double*) weight, bias, (const int*) trainData.target, (const double**) trainData.feature, trainSampleCnt, featureLen);
    double val_acc = getValAccuracy((const double*) alpha, (const double**) trainData.feature, (const double**) valData.feature, (const int*) valData.target, (const char*) kernel, trainSampleCnt, valSampleCnt, bias, featureLen, mDegree, c, sigma);
    printf("final train acc: %f | final val acc: %f \n", train_acc, val_acc);
    
    //6. free the memory
    //6.1 free kernelMatrix
    for (i = 0; i < trainSampleCnt; i++)
        free(kernelMatrix[i]);
    free(kernelMatrix);
    
    //6.2 free train data
    for (i = 0; i < trainSampleCnt; i++){
        free(trainData.feature[i]);
        //free(trainDataNorm.feature[i]);
    }
    free(trainData.feature);
    free(trainData.target);
    //free(trainDataNorm.feature);

    //6.3 free ValDataFeature
    for (i = 0; i < valSampleCnt; i++){
        free(valData.feature[i]);
        //free(valDataNorm.feature[i]);
    }
    free(valData.feature);
    free(valData.target);
    //free(valDataNorm.feature);

    //6.4 free A
    for (i=0; i < optMatrixLen; i++)
        free(A[i]);
    free(A);
    free(alpha);
    free(b);
    return 0;
}


double generateGuassRandom(){
    double u1, u2, w, res;
    u1 = (double)rand() / RAND_MAX;
    u2 = (double)rand() / RAND_MAX;
    double pai = acos(-1.0); 
    w = sqrt(-2.0 * log(u1)) * cos(2.0* pai *u2);
    res = (w+3)/6;
    return res > 1? 1 : res < 0 ? 0 : res;
}

#ifdef DEBUG_GEN_DATA
void getData(Data_p* data, char* Path, int featureLen, int sampleCnt){
   //get test feature and target
    data->feature = (double**)malloc(sampleCnt*sizeof(double*));
    data->target = (int*)malloc(sampleCnt*sizeof(int));
    int i,j;
    for (i=0; i<sampleCnt; i++){
        data->feature[i] = (double*)malloc(featureLen*sizeof(double));
        data->target[i] = rand()&1 ? 1: -1;
        for (j=0; j<featureLen; j++)
            data->feature[i][j] = generateGuassRandom(); 
    }
}
#endif

void getData(Data_p *data, char *path, int featureLen, int sampleCnt)
{
        FILE *file = fopen(path, "r");
        if (file == NULL)
        {
                perror("Error while opening the file.\n");
                exit(EXIT_FAILURE);
        }

        data->feature = (double **)malloc(sampleCnt * sizeof(double *));
        data->target = (int *)malloc(sampleCnt * sizeof(int));
        int i=0;
        int j=0;
        char line[MAX_LINE_LENGTH];
        while (fgets(line, MAX_LINE_LENGTH, file) && i < sampleCnt){
                j = 0;
                char* token = strtok(line, ",");
                data->feature[i] = (double *)malloc(featureLen * sizeof(double));
                while (token != NULL){
                        if (j<featureLen)
                                data->feature[i][j++] = atof(token);
                        else 
                                data->target[i] = atoi(token);
                        token = strtok(NULL, ",");
                }
                i++;
        }
        fclose(file);
}

int featureNorm(const double** trainDataFeature, const double** valDataFeature, double** trainDataNormFeature, double** valDataNormFeature, int featureLen, int trainSampleCnt, int valSampleCnt){
       int i,j;
       double minArr [featureLen];
       double maxArr [featureLen];
       for (i=0; i<featureLen; i++){
            minArr[i] = INT_MAX;
            maxArr[i] = INT_MIN;
       }
       for (i=0; i<featureLen; i++)
        for(j=0; j<trainSampleCnt; j++){
            if (trainDataFeature[j][i] < minArr[i])
                minArr[i] = trainDataFeature[j][i];
            if (trainDataFeature[j][i] > maxArr[i]) 
                maxArr[i] = trainDataFeature[j][i];
        }
    
       for (i=0; i<featureLen; i++){
        double dist = maxArr[i]-minArr[i];
        for (j=0; j<trainSampleCnt; j++)
            trainDataNormFeature[j][i] = (trainDataFeature[j][i] - minArr[i])/dist;
        for (j=0; j<valSampleCnt; j++)
            valDataNormFeature[j][i] = (valDataFeature[j][i] - minArr[i])/dist;
       }

       return 0;
}

double linearKernel(const double* x1, const double* x2, int featureLen, int mDegree, double c, double sigma){
        //K(x, y) = x ⋅ y
        double res = 0.0;
        int i;
        for (i=0; i<featureLen; i++)
                res += x1[i] * x2[i];
        return res;
}

double quadraticKernel(const double* x1, const double* x2, int featureLen, int mDegree, double c, double sigma){
   //K(x,y) = ( x ⋅ y + c)**2
   double res = 0.0;
   int i;
   for (i=0; i<featureLen; i++)
        res += x1[i] * x2[i];
   return pow(res + c, 2);
}

double polynomialKernel(const double* x1, const double* x2, int featureLen, int mDegree, double c, double sigma){
   //K(x, y) = (x ⋅ y + c)**d
   double innerProd = 0.0;
   int i;
   for (i=0; i<featureLen; i++)
       innerProd += x1[i] * x2[i];

   return pow(c + innerProd, mDegree);
}

double rbfKernel(const double* x1, const double* x2, int featureLen, int mDegree, double c, double sigma){
    // Radial Basis Function (RBF) kernel
    double squaredDistance = 0.0;
    int i;
    for (i=0; i < featureLen; i++)
        squaredDistance += pow(x1[i]-x2[i], 2);
    return exp(-1*squaredDistance/(2*sigma*sigma));
}

double computeObj(const double* alpha, const Data_p* trainData, const double** kernelMatrix, double beta, int trainSampleCnt){
    //04-handout.pdf: page 43-44
    double res_part1 = 0.0, res_part2 = 0.0, res_part3 = 0.0;
    int* target = trainData->target;
    int i,j;
    for (i=0; i<trainSampleCnt; i++){
        res_part1 += alpha[i];
        for(j=0; j<trainSampleCnt; j++)
            res_part2 += alpha[i] * alpha[j] * target[i] * target[j] * kernelMatrix[i][j];
        res_part3 += alpha[i] * target[i];
    }
    return res_part1 + -0.5 * res_part2 + -0.5 * beta * pow(res_part3, 2);
}
bool updateAlpha(double* alpha, const Data_p* trainData, const double** kernelMatrix, double beta, double learnRate, int trainSampleCnt, int C, double limit){
    bool is_stop = false;
    int i,j;
    int* target = trainData->target;
    double res_1, res_2;
    for (i=0; i<trainSampleCnt; i++) {
        res_1 = 0.0;
        for (j=0; j<trainSampleCnt; j++)
            res_1 += alpha[j] * target[i] * target[j] * kernelMatrix[i][j];
        
        res_2 = 0.0;
        for (j=0; j<trainSampleCnt; j++)
            res_2 += alpha[j] * target[i] * target[j];
        

        alpha[i] += learnRate * (1-res_1-beta*res_2);
        if (1-res_1-beta*res_2 < limit){
            is_stop = true;
            break;
        }
        //clip alpha
        alpha[i] = fmax(0, alpha[i]);
        alpha[i]  = fmin(C, alpha[i]);
    
    }
    return is_stop;
}

void updateBeta(double* beta, const double* alpha, const Data_p* trainData, int trainSampleCnt){
    double res = 0.0;
    int* target = trainData->target;
    int i;
    for (i=0; i<trainSampleCnt; i++)
        res += alpha[i] * target[i];

    *beta += 0.5 * pow(res, 2);
}

double getTrainAccuracyWithWeight(const double* weight, double bias, const int* dataTarget, const double** dataFeature, int sampleCnt, int featureLen){
    int i, j;
    int rightCnt = 0;
    for (i=0; i<sampleCnt; i++){
        double res = 0;
        for (j=0; j<featureLen; j++){
            res += weight[i] * dataFeature[i][j];
        }
        res += bias;
        if (res * dataTarget[i] > 0)
            rightCnt += 1;
    }
    return rightCnt/sampleCnt;
}

double getTrainAccuracy(const double* alpha, const int* dataTarget, const double** kernelMatrix, int sampleCnt, double bias){
    int rightCnt = 0;
    int i,j;
    for (i=0; i<sampleCnt; i++){
        double res = 0.0;
        for (j=0; j<sampleCnt; j++){
            res += alpha[j] * kernelMatrix[j][i];
        }
        res += bias;
        if (res*dataTarget[i] >0)
            rightCnt += 1;
    }
    return rightCnt/sampleCnt;
}

double getValAccuracy(const double* alpha, const double** trainDataFeature, const double** valDataFeature, const int* valDataTarget, const char* kernel, int trainSampleCnt, int valSampleCnt, double bias, int featureLen, int mDegree, double c, double sigma){
        int rightCnt = 0;
        int i, j;
        double (*kernelFunction)(const double*, const double*, int, int, double, double) = NULL;
        switch (kernel[0]) {
                case 'l': kernelFunction = linearKernel; break;
                case 'q': kernelFunction = quadraticKernel; break;
                case 'p': kernelFunction = polynomialKernel; break;
                case 'r': kernelFunction = rbfKernel; break;
                default: kernelFunction = linearKernel; break; // Handle default kernel type
        }
        for (i=0; i<valSampleCnt; i++){
                double res = 0.0;
                for (j=0; j<trainSampleCnt; j++){
                        res += alpha[j] * kernelFunction(trainDataFeature[j], valDataFeature[i], featureLen, mDegree, c, sigma);
                }
                res += bias;
                if (res*valDataTarget[i] > 0)
                        rightCnt += 1;
        }
        return rightCnt/valSampleCnt;
}

int conjugateDesc(const double** A, const double* b, double* x, int maxIte, double eps, int nSample){
    //1. initialize part 
    int i,j,k;
    double* r = (double *)malloc(nSample*sizeof(double));
    double* p = (double *)malloc(nSample*sizeof(double));
    for (i=0; i<nSample; i++){
        r[i] = b[i];
        p[i] = b[i];
    }

    double alpha = 0;
    double beta = 0;
    double rrPrev = 0;
    double rrCur = 0;
    for (i=0; i<nSample; i++)
        rrPrev += r[i] * r[i];
    //2.iteration part 
    for (k=0; k<maxIte; k++){
        //2.1 A * p_k
        double ap[nSample];
        for (i=0; i<nSample; i++){
            ap[i] = 0;
            for(j=0; j<nSample; j++){
                ap[i] += A[i][j] * p[j];
            }
        }
        //2.3 alpha_k
        double tmp = 0;
        for (i=0; i<nSample; i++){
            tmp += p[i] * ap[i];
        }
        alpha = rrPrev / tmp;
        //2.4 x_k+1 
        for (i=0; i<nSample; i++){
            x[i] = x[i] + (alpha * p[i]);           
        }
        //2.5 r_k+1
        for (i=0; i<nSample; i++){
            r[i] = r[i] - alpha * ap[i];
        }
        //2.6 beta_k
        //2.6.1 r_k+1^T * r_k+1
        rrCur = 0;
        for (i=0; i<nSample; i++){
            rrCur += r[i] * r[i];
        }
        //2.6.2 beta 
        beta = rrCur / rrPrev;
        rrPrev = rrCur;
        //2.7 p_k+1 
        for (i=0; i<nSample; i++){
            p[i] = r[i] + beta * p[i];
        }
        //if r_k+1 <= eps, stop iterating
        if (rrCur < eps){
            printf("rr_cur less than eps, break\n");
            break;
        }
    }
    return 0;
}

int svmToConjugate(const int* trainTarget, const double** kernelMatrix, double** A, double* b, int optMatrixLen, int C){
        int i,j;
        for (i=0; i<optMatrixLen; i++){
                b[i] = trainTarget[i] - trainTarget[optMatrixLen]; 
        }

        for (i=0; i<optMatrixLen; i++){
                for (j=0; j<optMatrixLen; j++){
                        A[i][j] = kernelMatrix[i][j] - kernelMatrix[j][optMatrixLen] - kernelMatrix[i][optMatrixLen] + kernelMatrix[optMatrixLen][optMatrixLen];             
                        if (i==j)
                            A[i][j] += 1/(double)C;
                }
        }
        return 0;
}

double getBias(const int* trainTarget, const double** kernelMatrix, const double* alpha, int optMatrixLen, int C){
    double tmp_2 = 0;
    double tmp_3 = 0;
    int i;
    for (i=0; i<optMatrixLen; i++){
        tmp_2 += alpha[i];
        tmp_3 += kernelMatrix[optMatrixLen][i] * alpha[i];
    }
    tmp_2 = tmp_2 * (kernelMatrix[optMatrixLen][optMatrixLen]+1/C);
 
    double res = trainTarget[optMatrixLen] + tmp_2 - tmp_3;
    return res;
}

int isSymMatrix(const double** A, int size){
    int isSym = 1;
    int i,j;
    for(i=0; i<size; i++){
        for(j=i+1; j<size; j++){
            if(!isSame(A[i][j],A[j][i]))
                 isSym = 0;              
        }
    }
    return isSym;
}


int isSame(double a, double b){
    return fabs(a-b) < 1e-9;
}

void printInSym(const double** A, int size){
    int i,j;
    for (i=0; i<size; i++)
        for(j=i+1; j<size; j++){
            if(!isSame(A[i][j],A[j][i]))
                printf("A[%d][%d] = %lf, A[%d][%d] = %lf\n", i, j, A[i][j], j, i, A[j][i]);
        }
}

void fromAlphaToWeight(const double* alpha, const double** trainDataFeature, double* weight, int sampleCnt, int featureLen){
    int i,j;
    for (i=0; i<featureLen; i++){
        for (j=0; j<sampleCnt; j++){
            weight[i] += alpha[i] * trainData.feature[i][j];  
            }
    }
}

int checkConjuResWithOrig(const double** kernelMatrix, const double* alpha, double bias, const int* target, int sampleCnt){
        int isSame = 1;
        int i,j;
        for(i=0; i<sampleCnt; i++){
                double pred = 0;
                for (j=0; j<sampleCnt; j++){
                        pred += kernelMatrix[i][j] * alpha[j];
                }
                pred += bias;
                if (pred != target[i]){
                        isSame = 0;
                        printf("sample %d: pred %f, target %d\n", i, pred, target[i]);
                }
        }
        return isSame;
}

int matrixMul(const double ** trainDataFeature, double** kernelMatrix, int trainSampleCnt){
        //trainDataFeature * transpose of (trainDataFeature)
        int i, j, k;
        for (i=0; i<trainSampleCnt; i++){
            for (j=0; j<trainSampleCnt; j++){
               kernelMatrix[i][j] = 0;
            for (k=0; k<trainSampleCnt; k++){
               kernelMatrix[i][j] += trainDataFeature[i][k] * trainDataFeature[j][k]; 
            }
          }
        }
        return 0;
}




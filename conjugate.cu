#include "svm.h"

int conjugateDescCPU(const float* A, const float* b, float* x, int maxIte, float eps, int nSample){
/*
conjugate descent algorithm sequential version for SVM optimization 
solves Ax = b problem 
A: nSample* nSample matrix 
x: nSample vector
b: nSample vector
maxIte : max iteration number 
eps : stopping criteria
nSample : number of input sample, 
*/
        //1. initialize part 
        int i,j,k;
        float* r = (float *)malloc(nSample*sizeof(float));
        float* p = (float *)malloc(nSample*sizeof(float));
        for (i=0; i<nSample; i++){
                r[i] = b[i];
                p[i] = b[i];
        }

        float alpha = 0;
        float beta = 0;
        float rrPrev = 0;
        float rrCur = 0;
        for (i=0; i<nSample; i++)
                rrPrev += r[i] * r[i];
        //2.iteration part 
        for (k=0; k<maxIte; k++){
                //2.1 A * p_k
                float ap[nSample];
                for (i=0; i<nSample; i++){
                        ap[i] = 0;
                        for(j=0; j<nSample; j++){
                                ap[i] += A[i*nSample+j] * p[j];
                        }
                        //printf("cpu ite %d ap[%d]:%f\n", k, i, ap[i]);
                        //printf("p[%d]:%f\n", i, p[i]);
                }
                //2.2 alpha_k
                float tmp = 0;
                for (i=0; i<nSample; i++){
                        tmp += p[i] * ap[i];
                }
                alpha = rrPrev / tmp;
                //printf("cpu ite %d alpha %f\n", k, alpha);
                //2.3 x_k+1 
                for (i=0; i<nSample; i++){
                        x[i] = x[i] + (alpha * p[i]);           
                        //printf("cpu ite %d x[%d]:%f\n", k, i, x[i]);
                }
                //2.4 r_k+1
                for (i=0; i<nSample; i++){
                        r[i] = r[i] - alpha * ap[i];
                        //printf("cpu ite %d r[%d]:%f\n", k, i, r[i]);
                }
                //2.5 beta_k
                //2.5.1 r_k+1^T * r_k+1
                rrCur = 0;
                for (i=0; i<nSample; i++){
                        rrCur += r[i] * r[i];
                }
                //2.5.2 beta 
                beta = rrCur / rrPrev;
                rrPrev = rrCur;
                //printf("cpu ite %d beta %f\n", k, beta);
                //printf("cpu ite %d rrPrev %f\n", k, rrPrev);

                //2.6 p_k+1 
                for (i=0; i<nSample; i++){
                        p[i] = r[i] + beta * p[i];
                        //printf("cpu ite %d p[%d]:%f\n", k, i, p[i]);
                }
                //if r_k+1 <= eps, stop iterating
                if (rrCur < eps){
                        printf("rr_cur less than eps, break. Ite %d\n", k);
                        break;
                }
        }
        return 0;
}

int conjugateDescGPU(const float* A, const float* b, float* x, int maxIte, float eps, int nSample, int dimBlock){
        /*
conjugate descent algorithm sequential version for SVM optimization 
solves Ax = b problem 
A: nSample* nSample matrix 
x: nSample vector
b: nSample vector
maxIte : max iteration number 
eps : stopping criteria
nSample : number of input sample, 
         */

        //1. device var
        float * A_d, * b_d, * x_d, * alpha_d, * beta_d, * rrPrev_d, * rrCur_d, * r_d, * p_d, *ap_d;
        int i;
        //2.malloc
        cudaMalloc((void **)&A_d, nSample*nSample*sizeof(float));
        if(!A_d){ 
                printf("conju cannot allocated A_d of %d elements\n", nSample*nSample);
                exit(1);
        }

        cudaMalloc((void **)&b_d, nSample*sizeof(float));
        if(!b_d){ 
                printf("conju cannot allocated b_d of %d elements\n", nSample);
                exit(1);
        } 

        cudaMalloc((void **)&x_d, nSample*sizeof(float));
        if(!x_d){ 
                printf("conju cannot allocated x_d of %d elements\n", nSample);
                exit(1);
        }
        cudaMemset(x_d, 0, nSample*sizeof(float));

        cudaMalloc((void **)&alpha_d, sizeof(float));
        if(!alpha_d){ 
                printf("conju cannot allocated alpha_d of %d elements\n", 1);
                exit(1);
        }
        cudaMemset(alpha_d, 0, sizeof(float));

        cudaMalloc((void **)&beta_d, sizeof(float));
        if(!beta_d){ 
                printf("conju cannot allocated beta_d of %d elements\n", 1);
                exit(1);
        }
        cudaMemset(beta_d, 0, sizeof(float));

        cudaMalloc((void **)&rrPrev_d, sizeof(float));
        if(!rrPrev_d){ 
                printf("conju cannot allocated rrPrev_d of %d elements\n", 1);
                exit(1);
        }

        cudaMalloc((void **)&rrCur_d, sizeof(float));
        if(!rrPrev_d){ 
                printf("conju cannot allocated rrPrev_d of %d elements\n", 1);
                exit(1);
        }
        cudaMemset(rrCur_d, 0, sizeof(float));

        cudaMalloc((void **)&r_d, nSample*sizeof(float));
        if(!r_d){ 
                printf("conju cannot allocated r_d of %d elements\n", nSample);
                exit(1);
        }

        cudaMalloc((void **)&p_d, nSample*sizeof(float));
        if(!p_d){ 
                printf("conju cannot allocated p_d of %d elements\n", nSample);
                exit(1);
        }

        cudaMalloc((void **)&ap_d, nSample*sizeof(float));
        if(!ap_d){ 
                printf("conju cannot allocated ap_d of %d elements\n", nSample);
                exit(1);
        }

        //3.set device 
        int gridNum = int(ceil((nSample*nSample)/float(dimBlock*dimBlock)));
        dim3 grid(gridNum, 1, 1);
        dim3 block(dimBlock, dimBlock, 1);

        float* rrPrev = (float*)calloc(1, sizeof(float));
        for (i=0; i<nSample; i++)
                rrPrev[0] += b[i] * b[i];

        //4. transfer data
        cudaMemset(ap_d, 0, nSample*sizeof(float));
        cudaMemcpy(A_d, A, nSample*nSample*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, nSample*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(rrPrev_d, rrPrev, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(r_d, b, nSample*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_d, b, nSample*sizeof(float), cudaMemcpyHostToDevice);

        //5.conjugate descent
        float* rrCur = (float*)calloc(1, sizeof(float));
        for(i=0; i<maxIte; i++){ 
                //compute ap
                cudaMemset(ap_d, 0, nSample*sizeof(float));
                cudaDeviceSynchronize();

                ConjugateDescDevForAp<<<grid, block>>>(A_d, p_d, ap_d, nSample);
                cudaDeviceSynchronize();

                //compute alpha
                cudaMemset(alpha_d, 0, sizeof(float));
                ConjugateDescDevForAlpha<<<grid, block>>>(ap_d, p_d, alpha_d, nSample);
                cudaDeviceSynchronize();
                ConjugateDescDevForAlphaFinal<<<grid, block>>>(alpha_d, rrPrev_d);
                cudaDeviceSynchronize();

                //compute x and r  
                ConjugateDescDevForXAndR<<<grid, block>>>(ap_d, p_d, alpha_d, r_d, x_d, nSample);
                cudaDeviceSynchronize();

                //compute beta
                cudaMemset(rrCur_d, 0, sizeof(float));
                cudaDeviceSynchronize();

                ConjugateDescDevForBeta<<<grid, block>>>(r_d, rrCur_d, nSample);
                cudaDeviceSynchronize();

                ConjugateDescDevForBetaFinal<<<grid, block>>>(rrCur_d, rrPrev_d, beta_d);
                cudaDeviceSynchronize();

                //compute p
                ConjugateDescDevForP<<<grid, block>>>(r_d, beta_d, p_d, nSample);
                cudaDeviceSynchronize();

                //check if matches stop criteria
                cudaMemcpy(rrCur, rrCur_d, sizeof(float), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                if (rrCur[0] < eps) {
                        printf("rr_cur less than eps, break. Ite %d\n", i);
                        break;
                }
        }
        //6.copy result back to host
        cudaMemcpy(x, x_d, nSample*sizeof(float), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        cudaFree(A_d);
        cudaFree(b_d);
        cudaFree(x_d);
        cudaFree(alpha_d);
        cudaFree(beta_d);
        cudaFree(rrPrev_d);
        cudaFree(rrCur_d);
        cudaFree(p_d);
        cudaFree(r_d);
        cudaFree(ap_d);
        free(rrCur);
        free(rrPrev);
        return 0;
}


__global__ void ConjugateDescDevForAp(float* A, float* p, float* ap, int n) {
        /*
           calculate A*p_k
         */
        int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        int j;
        if(index < n){
                for (j=0; j<n; j++){
                        ap[index] += A[index*n+j] * p[j];
                }
        }
}

__global__ void ConjugateDescDevForAlpha(float* ap, float* p, float* alpha, int n) {
        /*
           calculate alpha middle result
         */
        int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        if(index < n){
                atomicAdd(&alpha[0], p[index] * ap[index]);
        }
}

__global__ void ConjugateDescDevForAlphaFinal(float* alpha, float* rrPrev) {
        /*
           calculate alpha final result
         */
        int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        if(index < 1){
                alpha[0] = rrPrev[0]/alpha[0];
        }
}


__global__ void ConjugateDescDevForXAndR(float* ap, float* p, float* alpha, float* r, float* x, int n) {
        /*
           calculate x and r
         */
        int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        if(index < n){
                x[index] += alpha[0] * p[index];
                r[index] -= alpha[0] * ap[index];
        }
}


__global__ void ConjugateDescDevForBeta(float* r, float * rrCur, int n) {
        /*
           calculate bete middle result
         */
        int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        if(index < n){
                atomicAdd(&rrCur[0], r[index] * r[index]);
        }
}

__global__ void ConjugateDescDevForBetaFinal(float* rrCur, float * rrPrev, float* beta) {
        /*
           calculate bete final result
         */
        int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        if(index<1){
                beta[0] = rrCur[0] / rrPrev[0];
                rrPrev[0] = rrCur[0];
        }
}

__global__ void ConjugateDescDevForP(float* r, float * beta, float* p, int n) {
        /*
           calculate p result
         */
        int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        if(index < n){
                p[index] = r[index] + beta[0] * p[index];
        }     
}

int svmToConjugateCPU(const int* trainTarget, const float** kernelMatrix, float* A, int optMatrixLen, int C){
        /*
        sequential version of transfering from SVM objective (kernel matrix) 
        to conjugate descent opt objective A 
        */
        int i, j;
        for (i=0; i<optMatrixLen; i++){
                for (j=0; j<optMatrixLen; j++){
                        A[i*optMatrixLen+j] = kernelMatrix[i][j] - kernelMatrix[j][optMatrixLen] - kernelMatrix[i][optMatrixLen] + kernelMatrix[optMatrixLen][optMatrixLen];       
                        if (i==j)
                               A[i*optMatrixLen+j] += 1/(float)C;
                }
        }
        return 0;
}

int svmToConjugateGPU(const int* trainTarget, const float** kernelMatrix, float* A, int optMatrixLen, int C, int dimBlock){
        /*
           cuda version of transfering from SVM objective (kernel matrix) 
           to conjugate descent opt objective A 
         */
        int i;
        int matrixLen = optMatrixLen+1;

        int gridNum = int(ceil(float(optMatrixLen*optMatrixLen)/(dimBlock*dimBlock)));
        //printf("gridNum %d\n", gridNum);
        dim3 grid(gridNum, 1, 1);
        dim3 block(dimBlock, dimBlock, 1);

        float * kernel_mat_d, * A_d, * q_d;
        cudaMalloc((void **)&kernel_mat_d, matrixLen*matrixLen*sizeof(float));
        if(!kernel_mat_d){ 
                printf("SvmToConju cannot allocated kernel_mat_d of %d*%d elements\n", matrixLen, matrixLen);
                exit(1);
        }
        //A_d is (n-1)*(n-1)
        cudaMalloc((void **)&A_d, optMatrixLen*optMatrixLen*sizeof(float));
        if(!A_d){ 
                printf("SvmToConju cannot allocated A_d of %d*%d elements\n", optMatrixLen, optMatrixLen);
                exit(1);
        }
        //q_d is n
        cudaMalloc((void **)&q_d, matrixLen*sizeof(float));
        if(!q_d){ 
                printf("SvmToConju cannot allocated q_d of %d elements\n", matrixLen);
                exit(1);
        }

        for (i=0; i<matrixLen*matrixLen; i+=matrixLen){
                int rowNum = i/matrixLen;
                cudaMemcpy(kernel_mat_d+i, kernelMatrix[rowNum], matrixLen*sizeof(float), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(q_d, kernelMatrix[optMatrixLen], matrixLen*sizeof(float), cudaMemcpyHostToDevice);

        SvmToConjugateDev<<<grid, block>>>(kernel_mat_d, q_d, A_d, optMatrixLen, C);
        cudaMemcpy(A, A_d, optMatrixLen*optMatrixLen*sizeof(float), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        cudaFree(kernel_mat_d);
        cudaFree(A_d);
        cudaFree(q_d);

        return 0;
}

__global__  void SvmToConjugateDev(float* kernel_mat_d, float* q_d, float* A_d, int n, int C)
{
        /*
           primary function of transfering from SVM objective (kernel matrix) 
           to conjugate descent opt objective A 
         */
        int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        //line num
        int i = int(floorf(index/n));
        //col num
        int j = index - n * i;

        if (index < n*n){
                A_d[index] = kernel_mat_d[i*(n+1)+j] - q_d[i] - q_d[j] + q_d[n];
                if (i==j)
                        A_d[index] += 1/(float)C;
        }
}


float getBias(const int* trainTarget, const float** kernelMatrix, const float* alpha, int optMatrixLen, int C){
        /*
           get bias for conjugate descent algorithm
         */
        float tmp_2 = 0;
        float tmp_3 = 0;
        int i;
        for (i=0; i<optMatrixLen; i++){
                tmp_2 += alpha[i];
                tmp_3 += kernelMatrix[optMatrixLen][i] * alpha[i];
        }
        tmp_2 = tmp_2 * (kernelMatrix[optMatrixLen][optMatrixLen]+1/C);

        float res = trainTarget[optMatrixLen] + tmp_2 - tmp_3;
        return res;
}



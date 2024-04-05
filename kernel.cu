#include "svm.h"

float linearKernel(const float* x1, const float* x2, int featureLen, int mDegree, float c, float sigma){
        //K(x, y) = x ⋅ y
        float res = 0.0;
        int i;
        for (i=0; i<featureLen; i++)
                res += x1[i] * x2[i];
        return res;
}

float quadraticKernel(const float* x1, const float* x2, int featureLen, int mDegree, float c, float sigma){
   //K(x,y) = ( x ⋅ y + c)**2
   float res = 0.0;
   int i;
   for (i=0; i<featureLen; i++)
        res += x1[i] * x2[i];
   return pow(res + c, 2);
}

float polynomialKernel(const float* x1, const float* x2, int featureLen, int mDegree, float c, float sigma){
   //K(x, y) = (x ⋅ y + c)**d
   float innerProd = 0.0;
   int i;
   for (i=0; i<featureLen; i++)
       innerProd += x1[i] * x2[i];

   return pow(c + innerProd, mDegree);
}

float rbfKernel(const float* x1, const float* x2, int featureLen, int mDegree, float c, float sigma){
    // Radial Basis Function (RBF) kernel
    float squaredDistance = 0.0;
    int i;
    for (i=0; i < featureLen; i++)
        squaredDistance += pow(x1[i]-x2[i], 2);
    return exp(-1*squaredDistance/(2*sigma*sigma));
}


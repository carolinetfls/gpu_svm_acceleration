# GPU Acceleration for Accelerating Matrix Multiplication for SVM
This project explores the acceleration of Support Vector Machine (SVM) computations using GPU acceleration, focusing on optimizing the kernel matrix construction and the Quadratic Programming optimization process. The aim is to address computational bottlenecks present in traditional SVMs, making them more suitable for large datasets.

## Steps to run the code:

### Compile:

```
nvcc -lm -g main.cu kernel.cu conjugate.cu -o svm
gcc -lm -g main.c -o svm_cpu
```

### Run (Take polynomial kernel as example)

```
./svm -k p -t ./data/3000_2.csv -v ./data/3000_2.csv -l 2 -c 3000 -n 3000 -e 0.0001 -r 0.0001 -C 200 -m 3
```

### Parameters for the command line options:

- -k r specifies that you want to use the RBF kernel.
- -t and -v specify the paths to your training and validation datasets.
- -l is the length of the feature vectors.
- -c is the counts of training samples.
- -n is the counts of validation samples.
- -s is the sigma parameter for the RBF kernel.
- -e is the limit for the gradient descent.
- -C is the regularization parameter.
- -r is the learning rate.
- -m is degree for the Polynomial kernel

## SVM cuda optimization

### 2.1 SVM learning goals, traininig methods and prediction functions

(1) SVM training goals (kernel version)  
![WechatIMG2356](https://github.com/xp2083/nyu_23fall_gpu_project/assets/112786083/00e0e585-7aed-4d27-980f-31d6db224ad3)
(2) SVM training methods  
Stochastic Coordinate Ascent/Descent algorithm  
(3) prediction functions  
![WechatIMG2357](https://github.com/xp2083/nyu_23fall_gpu_project/assets/112786083/93ce2347-0d87-4269-8f2b-0877f6dc5104)  
(4)reference of SVM  
[04-handout.pdf](https://github.com/xp2083/nyu_23fall_gpu_project/files/13246045/04-handout.pdf)

### 2.2 c++ implementation reference

<https://github.com/koba-jon/svm_cpp/blob/main/Kernel-SVM/src/svm.cpp>

### 2.3 SVM sota (for GPU-wise optimization)

a survey -> https://dl.acm.org/doi/abs/10.1145/2638404.2638474?casa_token=4aW5f89DqbEAAAAA:RH8wnbk8EX7EmtN1gCy0eZw7TyJWLpWlKjTlSvJx7lYLIm1iQX1ibsdkLzcKsK7nM6T2ki2Hq0EWJQ  
 LS-SVM GPU version -> PLSSVM  
 paper: https://ieeexplore.ieee.org/abstract/document/9835379  
 github: https://github.com/SC-SGS/PLSSVM/tree/main

LS-SVM https://link.springer.com/article/10.1023/A:1018628609742

conjugate descent and kernel calculation https://github.com/SC-SGS/PLSSVM/blob/main/include/plssvm/backends/gpu_csvm.hpp

### 2.4 Another baseline: ThunderSVM

https://github.com/Xtra-Computing/thundersvm/tree/master

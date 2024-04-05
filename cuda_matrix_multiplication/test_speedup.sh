nvcc -Xcompiler -fopenmp -o speedup speedup.cu
./speedup 100 100 100
./speedup 200 200 200
./speedup 500 500 500
./speedup 2000 2000 2000
# ./speedup 10000 10000 10000
# ./speedup 20000 20000 20000

# ./speedup 1024 1024 1024
# ./speedup 8 8 8
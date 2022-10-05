CUDA_VERSION ?= 11.8

default: test_cublaslt.cpp
	/usr/local/cuda-$(CUDA_VERSION)/bin/nvcc -std=c++14 -g -lcublasLt -O2 test_cublaslt.cpp -o test_cublaslt

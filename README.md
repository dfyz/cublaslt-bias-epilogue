# cublaslt-bias-epilogue
A reproducer to show that cuBLASLt appears to apply bias epilogue incorrectly when multiplying a float32 vector by a float32 matrix and the output matrix has ld > rows

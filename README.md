This is a standalone program that only needs a CUDA installation to reproduce the problem.

  * The issue was first detected in CUDA 11.4, but earlier versions are possibly also affected.
  * The issue can also be reproduced in `nvidia-docker` (tested with `nvidia/cuda:11.7.0-devel-ubuntu18.04`).

Use the following command to compile the program with CUDA 11.8 (the latest version as of right now):
```
$ /usr/local/cuda-11.8/bin/nvcc -O2 test_cublaslt.cpp -o test_cublaslt -l cublasLt
```

An example of parameters that trigger the problem:
```
$ ./test_cublaslt 1 4 64 fp32
FAIL: c[0, 1] = 7.68
```

This means that we:
  * multiplied a `1x768` matrix filled with `0.1f` by a `768x4` matrix filled with `0.1f`
    (both input matrices have `ld == rows`)
  * stored the multiplication result in a `1x4` matrix with `ld == 64`
  * applied a bias consisting of `10.0f` to the result
  * **found `7.68` in one of the output elements**

However, all output elements should be equal to `17.68` (modulo floating-point precision),
since `0.1 * 0.1 + ... + 0.1 * 0.1 == 7.68` and `7.68 + 10.0 == 17.68`. It appears that
the bias wasn't applied to some of the output elements.

Any of the following makes the problem disappear:
   * increasing the number of rows of the left matrix (`./test_cublaslt 2 4 64 fp32`)
  * increasing the number of columns of the right matrix to a sufficiently large value
      (`./test_cublaslt 1 100 64 fp32`)
  * setting `ld == rows` in the output (`./test_cublaslt 1 4 1 fp32`)
  * using float16 instead of float32 (`./test_cublaslt 1 4 64 fp16`)

#include <stdio.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

static void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

static void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

template <typename T>
int Run(unsigned long long aRows, unsigned long long bCols, unsigned long long cLd, cublasComputeType_t computeType, cudaDataType_t dataType) {
    if (aRows > cLd) {
        printf("aRows should not exceed cLd\n");
        return 1;
    }

    cudaError_t cudaStat;
    cudaStat = cudaSetDevice(0);
    if (cudaStat != cudaSuccess) {
        printf("cudaSetDevice failed\n");
        return 1;
    }

    void* devPtr = nullptr;
    cublasLtHandle_t ltHandle = nullptr;
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t adesc = nullptr, bdesc = nullptr, cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    int exitCode = 0;
    try {
        size_t workspaceSize = 4 << 20;
        cudaStat = cudaMalloc(&devPtr, workspaceSize + (aRows*768 + bCols*768 + bCols*cLd + aRows) * sizeof(T));
        if (cudaStat != cudaSuccess) {
            printf("cudaMalloc failed\n");
            throw std::logic_error(":(");
        }
        void* workspace = devPtr;
        T* a = (T*)((char*)devPtr + workspaceSize);
        T* b = a + aRows*768;
        T* c = b + bCols*768;
        T* bias = c + bCols*cLd;
        const size_t maxDataSize = std::max(aRows, bCols)*768;
        std::vector<T> data(maxDataSize);
        for (size_t i = 0; i < maxDataSize; i++) {
            data[i] = 0.1f;
        }
        std::vector<T> biasData(aRows);
        for (size_t i = 0; i < aRows; ++i) {
            biasData[i] = 10.0f;
        }
        checkCudaStatus(cudaMemcpy(a, data.data(), aRows*768*sizeof(T), cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(b, data.data(), bCols*768*sizeof(T), cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(bias, biasData.data(), aRows*sizeof(T), cudaMemcpyHostToDevice));

        checkCublasStatus(cublasLtCreate(&ltHandle));
        checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, computeType, dataType));
        cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;
        checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

        checkCublasStatus(cublasLtMatrixLayoutCreate(&adesc, dataType, aRows, 768, aRows));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&bdesc, dataType, 768, bCols, 768));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&cdesc, dataType, aRows, bCols, cLd));

        checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
        checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

        int returnedResults = 0;
        checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, adesc, bdesc, cdesc, cdesc, preference, 1, &heuristicResult, &returnedResults));
        if (returnedResults == 0) {
            printf("no algorithms available\n");
            throw std::logic_error(":(");
        }

        checkCudaStatus(cudaDeviceSynchronize()); // not really necessary, just to be sure

        T alpha = 1.0f, beta = 0.0f;
        checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, &alpha, a, adesc, b, bdesc, &beta, c, cdesc, c, cdesc, &heuristicResult.algo, workspace, workspaceSize, 0));

        checkCudaStatus(cudaDeviceSynchronize()); // not really necessary, just to be sure

        checkCudaStatus(cudaMemcpy(data.data(), c, bCols*cLd*sizeof(T), cudaMemcpyDeviceToHost));
        const float reference = 17.68; // (768 * 0.1^2) + 10
        ssize_t firstBadRow = -1;
        ssize_t firstBadCol = -1;
        float firstBadVal{};
        bool allGood = true;
        for (size_t r = 0; allGood && r < aRows; ++r) {
            for (size_t c = 0; allGood && c < bCols; ++c) {
                float val = (float)data[cLd*c + r];
                if (std::abs(val - reference) > 0.5f) {
                    firstBadRow = r;
                    firstBadCol = c;
                    firstBadVal = val;
                    allGood = false;
                    break;
                }
            }
        }

        if (allGood) {
            puts("OK");
        } else {
            printf("FAIL: c[%zd, %zd] = %.2f\n", firstBadRow, firstBadCol, firstBadVal);
        }
    } catch (std::exception&) {
        exitCode = 1;
    }
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (cdesc) cublasLtMatrixLayoutDestroy(cdesc);
    if (bdesc) cublasLtMatrixLayoutDestroy(bdesc);
    if (adesc) cublasLtMatrixLayoutDestroy(adesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
    if (devPtr) cudaFree(devPtr);
    return exitCode;
}

int main(int argc, char* argv[])
{
    if (argc != 5) {
        printf("Usage: %s A_ROWS B_COLS C_LD fp16|fp32\n", argv[0]);
        return 1;
    }

    unsigned long long aRows = std::stoull(argv[1]);
    unsigned long long bCols = std::stoull(argv[2]);
    unsigned long long cLd = std::stoull(argv[3]);
    if (strcmp(argv[4], "fp16") == 0) {
        return Run<half>(aRows, bCols, cLd, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    } else if (strcmp(argv[4], "fp32") == 0) {
        return Run<float>(aRows, bCols, cLd, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    } else {
        printf("Unknown data type: %s\n", argv[2]);
        return 1;
    }
}

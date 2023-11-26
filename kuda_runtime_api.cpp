#include "kuda_runtime_api.h"
#include <jni.h>
#include <cuda_runtime_api.h>

//https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_getRuntimeVersion(JNIEnv * env, jobject instance) {
    int runtimeVersion;

    cudaError_t cudaStatus = cudaRuntimeGetVersion(&runtimeVersion);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return runtimeVersion;
}

JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_getDivice(JNIEnv* env, jobject instance) {
    int diviceCode;

    cudaError_t cudaStatus = cudaGetDevice(&diviceCode);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return diviceCode;
}
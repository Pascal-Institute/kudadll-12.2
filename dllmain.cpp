#include "sample.h"
#include <jni.h>
#include <cuda_runtime_api.h>

JNIEXPORT jint JNICALL Java_com_example_MyClass_nativeMethod(JNIEnv * env, jobject instance) {
    int cudaVersion;
    cudaError_t cudaStatus = cudaRuntimeGetVersion(&cudaVersion);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return cudaVersion;
}
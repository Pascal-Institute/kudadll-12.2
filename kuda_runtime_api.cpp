#include "kuda_runtime_api.h"
#include <jni.h>
#include <cuda_runtime_api.h>

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceGetLimit(JNIEnv* env, jobject obj, jbyte limit) {
    
    size_t pValue;
    
    cudaError_t cudaStatus = cudaDeviceGetLimit(&pValue, static_cast<cudaLimit>(limit));

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return pValue;
}

JNIEXPORT jstring JNICALL Java_kuda_RuntimeAPI_deviceGetPCIBusId(JNIEnv* env, jobject obj, jint device) {
    
    const int maxBufferLen = 13;
    char pciBusId[maxBufferLen];
    
    cudaError_t cudaStatus = cudaDeviceGetPCIBusId(pciBusId, maxBufferLen, device);

    if (cudaStatus != cudaSuccess) {
        return env->NewStringUTF("Error retrieving PCI Bus ID");
    }

    return env->NewStringUTF(pciBusId);
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceGetStreamPriorityRange(JNIEnv* env, jobject obj) {
    
    int leastPriority;
    int greatestPriority;

    cudaError_t cudaStatus = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return (leastPriority - greatestPriority);
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceSetCacheConfig(JNIEnv* env, jobject obj, jint cacheConfig) {

    cudaError_t cudaStatus = cudaDeviceSetCacheConfig(static_cast<cudaFuncCache>(cacheConfig));

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceSetLimit(JNIEnv* env, jobject obj, jbyte limit, jsize value) {
    
    cudaError_t cudaStatus = cudaDeviceSetLimit(static_cast<cudaLimit>(limit), (size_t) value);

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceSetSharedMemConfig(JNIEnv* env, jobject obj, jint config) {
    
    cudaError_t cudaStatus = cudaDeviceSetSharedMemConfig(static_cast<cudaSharedMemConfig>(config));

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceSynchronize(JNIEnv* env, jobject instance) {
    
    cudaError_t cudaStatus = cudaDeviceSynchronize();

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceReset(JNIEnv* env, jobject instance) {
    
    cudaError_t cudaStatus = cudaDeviceReset();

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_runtimeGetVersion(JNIEnv * env, jobject instance) {
    
    int runtimeVersion;

    cudaError_t cudaStatus = cudaRuntimeGetVersion(&runtimeVersion);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return runtimeVersion;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_getDevice(JNIEnv* env, jobject instance) {
   
    int diviceCode;

    cudaError_t cudaStatus = cudaGetDevice(&diviceCode);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return diviceCode;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_getDiviceCount(JNIEnv* env, jobject instance) {
    int diviceCount;

    cudaError_t cudaStatus = cudaGetDeviceCount(&diviceCount);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return diviceCount;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_initDevice(JNIEnv* env, jobject obj, jint device, jint deviceFlags, jint flags) {
    
    cudaError_t cudaStatus = cudaInitDevice((int)device, (unsigned int)deviceFlags, (unsigned int) flags);

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_lpcCloseMemHandle(JNIEnv* env, jobject instance, jlong devicePtr) {

    cudaError_t cudaStatus = cudaIpcCloseMemHandle((void*) devicePtr);

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_setDevice(JNIEnv* env, jobject instance, jint device) {
   
    cudaError_t cudaStatus = cudaSetDevice((int) device);

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_setDeviceFlags(JNIEnv* env, jobject instance, jint flags) {

    cudaError_t cudaStatus = cudaSetDeviceFlags((unsigned int) flags);

    return cudaStatus;
}

JNIEXPORT jstring JNICALL Java_kuda_RuntimeAPI_getErrorName(JNIEnv* env, jobject obj, jint error) {
    
    return env->NewStringUTF(cudaGetErrorName(static_cast<cudaError_t>(error)));
}

JNIEXPORT jstring JNICALL Java_kuda_RuntimeAPI_getErrorString(JNIEnv* env, jobject obj, jint error) {

    return env->NewStringUTF(cudaGetErrorString(static_cast<cudaError_t>(error)));
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_getLastError(JNIEnv* env, jobject obj) {

    cudaError_t cudaStatus = cudaGetLastError();

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_peekAtLastError(JNIEnv* env, jobject obj) {

    cudaError_t cudaStatus = cudaPeekAtLastError();

    return cudaStatus;
}

//6.4 Stream Management
JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_ctxResetPersistingL2Cache(JNIEnv* env, jobject obj) {
    
    cudaError_t cudaStatus = cudaCtxResetPersistingL2Cache();

    return cudaStatus;
}

JNIEXPORT jlong JNICALL Java_kuda_RuntimeAPI_streamCreate(JNIEnv* env, jobject obj) {

    cudaStream_t pStream;

    cudaError_t cudaStatus = cudaStreamCreate(&pStream);


    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return (jlong)pStream;
}

JNIEXPORT jlong JNICALL Java_kuda_RuntimeAPI_streamCreateWithFlags(JNIEnv* env, jobject obj, jint flags) {

    cudaStream_t pStream;

    cudaError_t cudaStatus = cudaStreamCreateWithFlags(&pStream, (unsigned int) flags);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return (jlong)pStream;
}


JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_eventCreate(JNIEnv* env, jobject obj, jobject event) {

    cudaError_t cudaStatus = cudaEventCreate(reinterpret_cast<cudaEvent_t*>(event));

    return cudaStatus;
}
#include "kuda_runtime_api.h"
#include <jni.h>
#include <cuda_runtime_api.h>

//6.1 Device Management
JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getLimit(JNIEnv* env, jclass cls, jbyte limit) {
    
    size_t pValue;
    
    cudaError_t cudaStatus = cudaDeviceGetLimit(&pValue, static_cast<cudaLimit>(limit));

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return pValue;
}

JNIEXPORT jstring JNICALL Java_kuda_runtimeapi_DeviceHandler_getPCIBusId(JNIEnv* env, jclass cls, jint device) {
    
    const int maxBufferLen = 13;
    char pciBusId[maxBufferLen];
    
    cudaError_t cudaStatus = cudaDeviceGetPCIBusId(pciBusId, maxBufferLen, device);

    if (cudaStatus != cudaSuccess) {
        return env->NewStringUTF("Error retrieving PCI Bus ID");
    }

    return env->NewStringUTF(pciBusId);
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getStreamPriorityRange(JNIEnv* env, jclass cls) {
    
    int leastPriority;
    int greatestPriority;

    cudaError_t cudaStatus = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return (leastPriority - greatestPriority);
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setCacheConfig(JNIEnv* env, jclass cls, jint cacheConfig) {

    cudaError_t cudaStatus = cudaDeviceSetCacheConfig(static_cast<cudaFuncCache>(cacheConfig));

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setLimit(JNIEnv* env, jclass cls, jbyte limit, jsize value) {
    
    cudaError_t cudaStatus = cudaDeviceSetLimit(static_cast<cudaLimit>(limit), (size_t) value);

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setSharedMemConfig(JNIEnv* env, jclass cls, jint config) {
    
    cudaError_t cudaStatus = cudaDeviceSetSharedMemConfig(static_cast<cudaSharedMemConfig>(config));

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_synchronize(JNIEnv* env, jclass cls) {
    
    cudaError_t cudaStatus = cudaDeviceSynchronize();

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_reset(JNIEnv* env, jclass cls) {
    
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

//cudaStreamAddCallback

//cudaStreamAttachMemAsync

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_streamBeginCapture(JNIEnv* env, jobject obj, jlong stream, jint mode) {

    CUstream_st* cudaStreamPointer = reinterpret_cast<CUstream_st*>(stream);

    cudaError_t cudaStatus = cudaStreamBeginCapture(cudaStreamPointer, static_cast<cudaStreamCaptureMode>(mode));

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_streamCopyAttributes(JNIEnv* env, jobject obj, jlong dst, jlong src) {

    CUstream_st* cudaDstStreamPointer = reinterpret_cast<CUstream_st*>(dst);

    CUstream_st* cudaSrcStreamPointer = reinterpret_cast<CUstream_st*>(src);

    cudaError_t cudaStatus = cudaStreamCopyAttributes(cudaDstStreamPointer, cudaSrcStreamPointer);

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

JNIEXPORT jlong JNICALL Java_kuda_RuntimeAPI_streamCreateWithPriority(JNIEnv* env, jobject obj, jint flags, jint priority) {

    cudaStream_t pStream;

    cudaError_t cudaStatus = cudaStreamCreateWithPriority(&pStream, (unsigned int) flags, (int) priority);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return (jlong)pStream;
}

JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_streamDestory(JNIEnv* env, jobject obj, jlong stream) {

    CUstream_st* cudaStreamPointer = reinterpret_cast<CUstream_st*>(stream);

    cudaError_t cudaStatus = cudaStreamDestroy(cudaStreamPointer);

    return cudaStatus;
}

//6.5 Event ManageMent
JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_EventHandler_create(JNIEnv* env, jclass cls) {

    cudaEvent_t event;

    cudaError_t cudaStatus = cudaEventCreate(&event);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return (jlong)event;
}

JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_EventHandler_createWithFlags(JNIEnv* env, jclass cls, jint flags) {
    
    cudaEvent_t event;

    cudaError_t cudaStatus = cudaEventCreateWithFlags(&event, (unsigned int)flags);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return (jlong)event;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_destroy(JNIEnv* env, jclass cls, jlong event) {
    
    CUevent_st* cudaEventPointer = reinterpret_cast<CUevent_st*>(event);

    cudaError_t cudaStatus = cudaEventDestroy(cudaEventPointer);

    return cudaStatus;
}

JNIEXPORT jfloat JNICALL Java_kuda_runtimeapi_EventHandler_elapsedTime(JNIEnv* env, jclass cls, jlong start, jlong end) {

    float ms;

    CUevent_st* cudaStartEventPointer = reinterpret_cast<CUevent_st*>(start);

    CUevent_st* cudaEndEventPointer = reinterpret_cast<CUevent_st*>(end);

    cudaError_t cudaStatus = cudaEventElapsedTime(&ms, cudaStartEventPointer, cudaEndEventPointer);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    return ms;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_query(JNIEnv* env, jclass cls, jlong event) {

    CUevent_st* cudaEventPointer = reinterpret_cast<CUevent_st*>(event);

    cudaError_t cudaStatus = cudaEventQuery(cudaEventPointer);

    return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_synchronize(JNIEnv* env, jclass cls, jlong event) {

    CUevent_st* cudaEventPointer = reinterpret_cast<CUevent_st*>(event);

    cudaError_t cudaStatus = cudaEventSynchronize(cudaEventPointer);

    return cudaStatus;
}
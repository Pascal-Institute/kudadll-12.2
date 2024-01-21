#include "kuda_driver_api.h"
#include <jni.h>
#include <cuda.h>

//2. Error Handling
JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_getErrorName(JNIEnv* env, jobject obj, jint error) {
	
	const char* pStr;

	CUresult cudaStatus = cuGetErrorName(static_cast<CUresult>(error), &pStr);
	
	jstring javaString = env->NewStringUTF(pStr);

	return javaString;
}

JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_getErrorString(JNIEnv* env, jobject obj, jint error) {

	const char* pStr;

	CUresult cudaStatus = cuGetErrorString(static_cast<CUresult>(error), &pStr);

	jstring javaString = env->NewStringUTF(pStr);

	return javaString;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_init(JNIEnv* env, jobject obj, jint flags) {
	
	CUresult cudaStatus = cuInit((unsigned int) flags);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_getDriverVersion(JNIEnv* env, jobject obj) {
	
	int driverVersion;

	CUresult cudaStatus = cuDriverGetVersion(&driverVersion);

	if (cudaStatus != CUDA_SUCCESS) {
		return cudaStatus;
	}

	return driverVersion;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_deviceGet(JNIEnv* env, jobject obj, jint ordinal) {

	CUdevice device;

	CUresult cudaStatus = cuDeviceGet(&device, ordinal);

	if (cudaStatus != CUDA_SUCCESS) {
		return cudaStatus;
	}

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_deviceGetCount(JNIEnv* env, jobject obj) {
	
	int count;

	CUresult cudaStatus = cuDeviceGetCount(&count);

	if (cudaStatus != CUDA_SUCCESS) {
		return cudaStatus;
	}

	return count;
}

//7. Context Management
//CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)
JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxRelease(JNIEnv* env, jobject obj, jint dev) {
	
	CUresult cudaStatus = cuDevicePrimaryCtxRelease(dev);

	return cudaStatus;
}
//CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
//CUresult cuDevicePrimaryCtxReset(CUdevice dev)
//CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev)
//CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int  flags)
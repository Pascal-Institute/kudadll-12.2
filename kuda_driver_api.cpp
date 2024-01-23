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

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_driverGetVersion(JNIEnv* env, jobject obj) {
	
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

//7. Primary Context Management

//CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxRelease(JNIEnv* env, jobject obj, jint dev) {
	
	CUresult cudaStatus = cuDevicePrimaryCtxRelease(dev);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxReset(JNIEnv* env, jobject obj, jint dev) {

	CUresult cudaStatus = cuDevicePrimaryCtxReset(dev);

	return cudaStatus;
}

//CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxSetFlags(JNIEnv* env, jobject obj, jint dev, jint flags) {

	CUresult cudaStatus = cuDevicePrimaryCtxSetFlags(dev, (unsigned int)flags);

	return cudaStatus;
}

//8. Context Management
//CUresult cuCtxCreate(CUcontext* pctx, unsigned int  flags, CUdevice dev)
//CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int  numParams, unsigned int  flags, CUdevice dev)
//CUresult cuCtxDestroy(CUcontext ctx)
//CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetCacheConfig(JNIEnv* env, jobject obj, jboolean dummy) {
	
	CUfunc_cache pconfig;

	CUresult cudaStatus = cuCtxGetCacheConfig(&pconfig);

	if (cudaStatus != CUDA_SUCCESS) {
		return cudaStatus;
	}

	return static_cast<int>(pconfig);
}

//CUresult cuCtxGetCurrent(CUcontext * pctx)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetDevice(JNIEnv* env, jobject obj) {
	
	CUdevice device;

	CUresult cudaStatus = cuCtxGetDevice(&device);

	if (cudaStatus != CUDA_SUCCESS) {
		return cudaStatus;
	}

	return (jint)device;
}

//CUresult cuCtxGetExecAffinity(CUexecAffinityParam * pExecAffinity, CUexecAffinityType type)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetFlags(JNIEnv* env, jobject obj) {
	unsigned int flags;

	CUresult cudaStatus = cuCtxGetFlags(&flags);

	if (cudaStatus != CUDA_SUCCESS) {
		return cudaStatus;
	}
	
	return (jint)flags;
}
//CUresult cuCtxGetId(CUcontext ctx, unsigned long long* ctxId)
//CUresult cuCtxGetLimit(size_t * pvalue, CUlimit limit)
//CUresult cuCtxGetSharedMemConfig(CUsharedconfig * pConfig)
//CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
//CUresult cuCtxPopCurrent(CUcontext * pctx)
//CUresult cuCtxPushCurrent(CUcontext ctx)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxResetPersistingL2Cache(JNIEnv* env, jobject obj) {

	CUresult cudaStatus = cuCtxResetPersistingL2Cache();

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetCacheConfig(JNIEnv* env, jobject obj, jint config) {
	
	CUresult cudaStatus = cuCtxSetCacheConfig(static_cast<CUfunc_cache>(config));
	
	return cudaStatus;
}

//CUresult cuCtxSetCurrent(CUcontext ctx)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetFlags(JNIEnv* env, jobject obj, jint flags) {
	
	CUresult cudaStatus = cuCtxSetFlags((unsigned int) flags);
	
	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetLimit(JNIEnv* env, jobject obj, jbyte limit, jsize value) {
	CUresult cudaStatus = cuCtxSetLimit(static_cast<CUlimit>(limit), value);
	
	return cudaStatus;
}

//CUresult cuCtxSetSharedMemConfig(CUsharedconfig config)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSynchronize(JNIEnv* env, jobject obj) {

	CUresult cudaStatus = cuCtxSynchronize();

	return cudaStatus;
}

//9. Context Management (DEPRECATED)

//10. Module Management
//CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues)
//CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path, unsigned int  numOptions, CUjit_option* options, void** optionValues)
//CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut)
//CUresult cuLinkCreate(unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_linkDestory(JNIEnv* env, jobject obj, jlong state) {
	
	CUlinkState cuLinkState = reinterpret_cast<CUlinkState>(state);

	CUresult cudaStatus = cuLinkDestroy(cuLinkState);

	return cudaStatus;
}

//CUresult cuModuleGetFunction(CUfunction * hfunc, CUmodule hmod, const char* name)
//CUresult cuModuleGetGlobal(CUdeviceptr * dptr, size_t * bytes, CUmodule hmod, const char* name)
//CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode * mode)
//CUresult cuModuleLoad(CUmodule * module, const char* fname)
//CUresult cuModuleLoadData(CUmodule * module, const void* image)
//CUresult cuModuleLoadDataEx(CUmodule * module, const void* image, unsigned int  numOptions, CUjit_option * options, void** optionValues)
//CUresult cuModuleLoadFatBinary(CUmodule * module, const void* fatCubin)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_moduleUnload(JNIEnv* env, jobject obj, jlong hmod) {

	CUmodule cuModule = reinterpret_cast<CUmodule>(hmod);

	CUresult cudaStatus = cuModuleUnload(cuModule);

	return cudaStatus;
}
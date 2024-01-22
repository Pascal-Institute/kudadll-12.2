#include <jni.h>

//https://docs.nvidia.com/cuda/cuda-driver-api/index.html

#ifdef __cplusplus
extern "C" {
#endif

	//2. Error Handling
	JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_getErrorName(JNIEnv* env, jobject obj, jint error);

	JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_getErrorString(JNIEnv* env, jobject obj, jint error);

	//3. Initialization
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_init(JNIEnv* env, jobject obj, jint flags);

	//4. Version Management
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_driverGetVersion(JNIEnv* env, jobject obj);

	//5. Device Management
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_deviceGet(JNIEnv* env, jobject obj, jint ordinal);

	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_deviceGetCount(JNIEnv* env, jobject obj);

	//7. Context Management
	//CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)
	
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxRelease(JNIEnv* env, jobject obj, jint dev);
	
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxReset(JNIEnv* env, jobject obj, jint dev);
	
	//CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev)
	
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxSetFlags(JNIEnv* env, jobject obj, jint dev, jint flags);
	
	//8. Context Management
	//CUresult cuCtxCreate(CUcontext* pctx, unsigned int  flags, CUdevice dev)
	//CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int  numParams, unsigned int  flags, CUdevice dev)
	//CUresult cuCtxDestroy(CUcontext ctx)
	//CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version)
	//CUresult cuCtxGetCacheConfig(CUfunc_cache * pconfig)
	//CUresult cuCtxGetCurrent(CUcontext * pctx)
	//CUresult cuCtxGetDevice(CUdevice * device)
	//CUresult cuCtxGetExecAffinity(CUexecAffinityParam * pExecAffinity, CUexecAffinityType type)
	//CUresult cuCtxGetFlags(unsigned int* flags)
	//CUresult cuCtxGetId(CUcontext ctx, unsigned long long* ctxId)
	//CUresult cuCtxGetLimit(size_t * pvalue, CUlimit limit)
	//CUresult cuCtxGetSharedMemConfig(CUsharedconfig * pConfig)
	//CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
	//CUresult cuCtxPopCurrent(CUcontext * pctx)
	//CUresult cuCtxPushCurrent(CUcontext ctx)
	//CUresult cuCtxResetPersistingL2Cache(void)
	//CUresult cuCtxSetCacheConfig(CUfunc_cache config)
	//CUresult cuCtxSetCurrent(CUcontext ctx)
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetFlags(JNIEnv* env, jobject obj, jint flags);
	
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetLimit(JNIEnv* env, jobject obj, jbyte limit, jsize value);

	//CUresult cuCtxSetSharedMemConfig(CUsharedconfig config)
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSynchronize(JNIEnv* env, jobject obj);

#ifdef __cplusplus
}
#endif
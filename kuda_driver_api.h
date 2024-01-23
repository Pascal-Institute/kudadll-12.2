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

	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetCacheConfig(JNIEnv* env, jobject obj, jboolean dummy);
	
	//CUresult cuCtxGetCurrent(CUcontext * pctx)

	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetDevice(JNIEnv* env, jobject obj);
	
	//CUresult cuCtxGetExecAffinity(CUexecAffinityParam * pExecAffinity, CUexecAffinityType type)
	
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetFlags(JNIEnv* env, jobject obj);
	
	//CUresult cuCtxGetId(CUcontext ctx, unsigned long long* ctxId)
	//CUresult cuCtxGetLimit(size_t * pvalue, CUlimit limit)
	//CUresult cuCtxGetSharedMemConfig(CUsharedconfig * pConfig)
	//CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
	//CUresult cuCtxPopCurrent(CUcontext * pctx)
	//CUresult cuCtxPushCurrent(CUcontext ctx)

	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxResetPersistingL2Cache(JNIEnv* env, jobject obj);
	
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetCacheConfig(JNIEnv* env, jobject obj, jint config);

	//CUresult cuCtxSetCurrent(CUcontext ctx)
	
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetFlags(JNIEnv* env, jobject obj, jint flags);
	
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetLimit(JNIEnv* env, jobject obj, jbyte limit, jsize value);

	//CUresult cuCtxSetSharedMemConfig(CUsharedconfig config)

	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSynchronize(JNIEnv* env, jobject obj);

	//9. Context Management (DEPRECATED)

	//10. Module Management
	//CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues)
	//CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path, unsigned int  numOptions, CUjit_option* options, void** optionValues)
	//CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut)
	//CUresult cuLinkCreate(unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut)

	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_linkDestroy(JNIEnv* env, jobject obj, jlong state);

	//CUresult cuModuleGetFunction(CUfunction * hfunc, CUmodule hmod, const char* name)
	//CUresult cuModuleGetGlobal(CUdeviceptr * dptr, size_t * bytes, CUmodule hmod, const char* name)
	//CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode * mode)
	//CUresult cuModuleLoad(CUmodule * module, const char* fname)
	//CUresult cuModuleLoadData(CUmodule * module, const void* image)
	//CUresult cuModuleLoadDataEx(CUmodule * module, const void* image, unsigned int  numOptions, CUjit_option * options, void** optionValues)
	//CUresult cuModuleLoadFatBinary(CUmodule * module, const void* fatCubin)

	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_moduleUnload(JNIEnv* env, jobject obj, jlong hmod);

	//11. Module Management (DEPRECATED)

	//12. Library Management
	//CUresult cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev)
	//CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel)
	//CUresult cuKernelGetName(const char** name, CUkernel hfunc)
	//CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int  val, CUkernel kernel, CUdevice dev)
	//CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev)
	//CUresult cuLibraryGetGlobal(CUdeviceptr * dptr, size_t * bytes, CUlibrary library, const char* name)
	//CUresult cuLibraryGetKernel(CUkernel * pKernel, CUlibrary library, const char* name)
	//CUresult cuLibraryGetManaged(CUdeviceptr * dptr, size_t * bytes, CUlibrary library, const char* name)
	//CUresult cuLibraryGetModule(CUmodule * pMod, CUlibrary library)
	//CUresult cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol)
	//CUresult cuLibraryLoadData(CUlibrary * library, const void* code, CUjit_option * jitOptions, void** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void** libraryOptionValues, unsigned int  numLibraryOptions)
	//CUresult cuLibraryLoadFromFile(CUlibrary * library, const char* fileName, CUjit_option * jitOptions, void** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void** libraryOptionValues, unsigned int  numLibraryOptions)
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_libraryUnload(JNIEnv* env, jobject obj, jlong library);

#ifdef __cplusplus
}
#endif
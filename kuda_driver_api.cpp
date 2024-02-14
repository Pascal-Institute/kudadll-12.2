#include "kuda_driver_api.h"
#include <jni.h>
#include <cuda.h>

//2. Error Handling
JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_getErrorName(JNIEnv* env, jobject obj, jint error) {
	
	const char* pStr;

	CUresult cuResult = cuGetErrorName(static_cast<CUresult>(error), &pStr);
	
	jstring javaString = env->NewStringUTF(pStr);

	return javaString;
}

JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_getErrorString(JNIEnv* env, jobject obj, jint error) {

	const char* pStr;

	CUresult cuResult = cuGetErrorString(static_cast<CUresult>(error), &pStr);

	jstring javaString = env->NewStringUTF(pStr);

	return javaString;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_init(JNIEnv* env, jobject obj, jint flags) {
	
	CUresult cuResult = cuInit((unsigned int) flags);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_driverGetVersion(JNIEnv* env, jobject obj) {
	
	int driverVersion;

	CUresult cuResult = cuDriverGetVersion(&driverVersion);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return driverVersion;
}

// 5. Device Management //

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_deviceGet(JNIEnv* env, jobject obj, jint ordinal) {

	CUdevice device;

	CUresult cuResult = cuDeviceGet(&device, ordinal);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return device;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_deviceGetCount(JNIEnv* env, jobject obj) {
	
	int count;

	CUresult cuResult = cuDeviceGetCount(&count);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return count;
}

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_deviceGetDefaultMemPool(JNIEnv* env, jobject obj, jint dev) {

	CUmemoryPool cuMemoryPool;

	CUresult cuResult = cuDeviceGetDefaultMemPool(&cuMemoryPool, dev);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuMemoryPool;
}

//CUresult cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType type, CUdevice dev)
//CUresult cuDeviceGetLuid ( char* luid, unsigned int* deviceNodeMask, CUdevice dev )
 
JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_deviceGetMemPool(JNIEnv* env, jobject obj, jint dev) {

	CUmemoryPool cuMemoryPool;

	CUresult cuResult = cuDeviceGetMemPool(&cuMemoryPool, dev);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuMemoryPool;
}

JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_deviceGetName(JNIEnv* env, jobject obj, jint len, jint dev) {
	
	char* name = new char[len + 1];

	CUresult cuResult = cuDeviceGetName(name, len, dev);

	jstring javaString = env->NewStringUTF(name);

	delete[] name;

	return javaString;
}

//7. Primary Context Management // COMPLETE

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxGetState(JNIEnv* env, jobject obj, jint dev) {
	
	unsigned int flags;

	int active;

	CUresult cuResult = cuDevicePrimaryCtxGetState(dev, &flags, &active);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return active;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxRelease(JNIEnv* env, jobject obj, jint dev) {
	
	CUresult cuResult = cuDevicePrimaryCtxRelease(dev);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxReset(JNIEnv* env, jobject obj, jint dev) {

	CUresult cuResult = cuDevicePrimaryCtxReset(dev);

	return cuResult;
}

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxRetain(JNIEnv* env, jobject obj, jint dev) {

	CUcontext pctx;

	CUresult cuResult = cuDevicePrimaryCtxRetain(&pctx, (CUdevice)dev);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)pctx;

}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxSetFlags(JNIEnv* env, jobject obj, jint dev, jint flags) {

	CUresult cuResult = cuDevicePrimaryCtxSetFlags(dev, (unsigned int)flags);

	return cuResult;
}

//8. Context Management//

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_ctxCreate(JNIEnv* env, jobject obj, jint flags, jint dev) {

	CUcontext cuContext;

	CUresult cuResult = cuCtxCreate(&cuContext, (unsigned int)flags, dev);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuContext;
}

//CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int  numParams, unsigned int  flags, CUdevice dev)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxDestroy(JNIEnv* env, jobject obj, jlong ctx){
	
	CUcontext cuContext = reinterpret_cast<CUcontext>(ctx);

	CUresult cuResult = cuCtxDestroy(cuContext);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetApiVersion(JNIEnv* env, jobject obj, jint ctx) {
	
	unsigned int version;

	CUcontext cuContext = reinterpret_cast<CUcontext>(ctx);

	CUresult cuResult = cuCtxGetApiVersion(cuContext, &version);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return version;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetCacheConfig(JNIEnv* env, jobject obj, jboolean dummy) {
	
	CUfunc_cache pConfig;

	CUresult cuResult = cuCtxGetCacheConfig(&pConfig);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return static_cast<int>(pConfig);
}

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_ctxGetCurrent(JNIEnv* env, jobject obj) {
	CUcontext pctx;

	CUresult cuResult = cuCtxGetCurrent(&pctx);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)pctx;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetDevice(JNIEnv* env, jobject obj) {
	
	CUdevice device;

	CUresult cuResult = cuCtxGetDevice(&device);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jint)device;
}

//CUresult cuCtxGetExecAffinity(CUexecAffinityParam * pExecAffinity, CUexecAffinityType type)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetFlags(JNIEnv* env, jobject obj) {
	unsigned int flags;

	CUresult cuResult = cuCtxGetFlags(&flags);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}
	
	return (jint)flags;
}

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_ctxGetId(JNIEnv* env, jobject obj, jlong ctx) {

	unsigned long long ctxId;

	CUcontext cuContext = reinterpret_cast<CUcontext>(ctx);

	CUresult cuResult = cuCtxGetId(cuContext, &ctxId);
	
	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)ctxId;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetLimit(JNIEnv* env, jobject obj, jint limit) {

	size_t pValue;

	CUlimit cuLimit = static_cast<CUlimit>(limit);

	CUresult cuResult = cuCtxGetLimit(&pValue, cuLimit);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jint)pValue;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxGetSharedMemConfig(JNIEnv* env, jobject obj, jboolean dummy) {
	
	CUsharedconfig pConfig;

	CUresult cuResult = cuCtxGetSharedMemConfig(&pConfig);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return static_cast<int>(pConfig);
}

JNIEXPORT jintArray JNICALL Java_kuda_driverapi_DriverAPI_ctxGetStreamPriorityRange(JNIEnv* env, jobject obj) {
	
	int leastPriority;

	int greatestPriority;

	jintArray result = env->NewIntArray(2);
	jint value[2];

	CUresult cuResult = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);

	if (cuResult != CUDA_SUCCESS) {
		
		jintArray failResult = env->NewIntArray(1);
		jint failValue[1];

		result = env->NewIntArray(1);

		value[1] = cuResult;

		env->SetIntArrayRegion(result, 0, 1, value);
		
		return result;
	}

	value[0] = leastPriority;
	value[1] = greatestPriority;

	env->SetIntArrayRegion(result, 0, 2, value);
	
	return result;
}

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_ctxPopCurrent(JNIEnv* env, jobject obj) {
	
	CUcontext pctx;

	CUresult cuResult = cuCtxPopCurrent(&pctx);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)pctx;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxPushCurrent(JNIEnv* env, jobject obj, jlong ctx) {

	CUcontext cuContext = reinterpret_cast<CUcontext>(ctx);

	CUresult cuResult = cuCtxPushCurrent(cuContext);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxResetPersistingL2Cache(JNIEnv* env, jobject obj) {

	CUresult cuResult = cuCtxResetPersistingL2Cache();

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetCacheConfig(JNIEnv* env, jobject obj, jint config) {
	
	CUresult cuResult = cuCtxSetCacheConfig(static_cast<CUfunc_cache>(config));
	
	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetCurrent(JNIEnv* env, jobject obj, jlong ctx) {

	CUcontext cuContext = reinterpret_cast<CUcontext>(ctx);

	CUresult cuResult = cuCtxSetCurrent(cuContext);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetFlags(JNIEnv* env, jobject obj, jint flags) {
	
	CUresult cuResult = cuCtxSetFlags((unsigned int) flags);
	
	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetLimit(JNIEnv* env, jobject obj, jbyte limit, jsize value) {
	
	CUresult cuResult = cuCtxSetLimit(static_cast<CUlimit>(limit), value);
	
	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSetSharedMemConfig(JNIEnv* env, jobject obj, jint config){
	
	CUresult cuResult = cuCtxSetSharedMemConfig(static_cast<CUsharedconfig>(config));

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxSynchronize(JNIEnv* env, jobject obj) {

	CUresult cuResult = cuCtxSynchronize();

	return cuResult;
}

//9. Context Management (DEPRECATED)

//10. Module Management
//CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues)
//CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path, unsigned int  numOptions, CUjit_option* options, void** optionValues)
//CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut)
//CUresult cuLinkCreate(unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_linkDestroy(JNIEnv* env, jobject obj, jlong state) {
	
	CUlinkState cuLinkState = reinterpret_cast<CUlinkState>(state);

	CUresult cuResult = cuLinkDestroy(cuLinkState);

	return cuResult;
}

//CUresult cuModuleGetFunction(CUfunction * hfunc, CUmodule hmod, const char* name)
//CUresult cuModuleGetGlobal(CUdeviceptr * dptr, size_t * bytes, CUmodule hmod, const char* name)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_moduleGetLoadingMode(JNIEnv* env, jobject obj, jboolean dummy) {

	CUmoduleLoadingMode cuLoadingModuleMode;

	CUresult cuResult = cuModuleGetLoadingMode(&cuLoadingModuleMode);
	
	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return static_cast<int>(cuLoadingModuleMode);
}

//CUresult cuModuleLoad(CUmodule * module, const char* fname)
//CUresult cuModuleLoadData(CUmodule * module, const void* image)
//CUresult cuModuleLoadDataEx(CUmodule * module, const void* image, unsigned int  numOptions, CUjit_option * options, void** optionValues)
//CUresult cuModuleLoadFatBinary(CUmodule * module, const void* fatCubin)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_moduleUnload(JNIEnv* env, jobject obj, jlong hmod) {

	CUmodule cuModule = reinterpret_cast<CUmodule>(hmod);

	CUresult cuResult = cuModuleUnload(cuModule);

	return cuResult;
}

//11. Module Management (DEPRECATED)

//12. Library Management
//CUresult cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev)

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_kernelGetFunction(JNIEnv* env, jobject obj, jlong kernel) {
	
	CUfunction pFunc;

	CUkernel cuKernel = reinterpret_cast<CUkernel>(kernel);

	CUresult cuResult = cuKernelGetFunction(&pFunc, cuKernel);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)pFunc;
}

//CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int  val, CUkernel kernel, CUdevice dev)
//CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev)
//CUresult cuLibraryGetGlobal(CUdeviceptr * dptr, size_t * bytes, CUlibrary library, const char* name)
//CUresult cuLibraryGetKernel(CUkernel * pKernel, CUlibrary library, const char* name)
//CUresult cuLibraryGetManaged(CUdeviceptr * dptr, size_t * bytes, CUlibrary library, const char* name)

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_libraryGetModule(JNIEnv* env, jobject obj, jlong library) {

	CUmodule cuModule;

	CUlibrary cuLibrary = reinterpret_cast<CUlibrary>(library);

	CUresult cuResult = cuLibraryGetModule(&cuModule, cuLibrary);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuModule;
}

//CUresult cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol)
//CUresult cuLibraryLoadData(CUlibrary * library, const void* code, CUjit_option * jitOptions, void** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void** libraryOptionValues, unsigned int  numLibraryOptions)
//CUresult cuLibraryLoadFromFile(CUlibrary * library, const char* fileName, CUjit_option * jitOptions, void** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void** libraryOptionValues, unsigned int  numLibraryOptions)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_libraryUnload(JNIEnv* env, jobject obj, jlong library) {
	
	CUlibrary cuLibrary = reinterpret_cast<CUlibrary>(library);

	CUresult cuResult = cuLibraryUnload(cuLibrary);

	return cuResult;
}

//13. Memory Management
//CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
//CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR * pArrayDescriptor, CUarray hArray)
//CUresult cuArrayCreate(CUarray * pHandle, const CUDA_ARRAY_DESCRIPTOR * pAllocateArray)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_arrayDestroy(JNIEnv* env, jobject obj, jlong hArray) {
	
	CUarray cuArray = reinterpret_cast<CUarray>(hArray);
	
	CUresult cuResult = cuArrayDestroy(cuArray);

	return cuResult;
}

//CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR * pArrayDescriptor, CUarray hArray)
//CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUarray array, CUdevice device)
//CUresult cuArrayGetPlane(CUarray * pPlaneArray, CUarray hArray, unsigned int  planeIdx)
//CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUarray array)

JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_deviceGetByPCIBusId(JNIEnv* env, jobject obj) {

	const char* pciBusId = "";

	CUdevice dev;

	CUresult cuResult = cuDeviceGetByPCIBusId(&dev, pciBusId);

	jstring javaString = env->NewStringUTF(pciBusId);

	return javaString;
}

JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_deviceGetPCIBusId(JNIEnv* env, jobject obj, jint len, jint dev) {

	char* pciBusId = (char*)malloc(sizeof(char) * (len + 1));

	CUresult cuResult = cuDeviceGetPCIBusId(pciBusId, len, (CUdevice) dev);

	jstring javaString = env->NewStringUTF(pciBusId);

	return javaString;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ipcCloseMemHandle(JNIEnv* env, jobject obj, jlong dptr) {

	CUresult cuResult = cuIpcCloseMemHandle(dptr);

	return cuResult;
}

//CUresult cuIpcGetEventHandle(CUipcEventHandle * pHandle, CUevent event)
//CUresult cuIpcGetMemHandle(CUipcMemHandle * pHandle, CUdeviceptr dptr)
//CUresult cuIpcOpenEventHandle(CUevent * phEvent, CUipcEventHandle handle)
//CUresult cuIpcOpenMemHandle(CUdeviceptr * pdptr, CUipcMemHandle handle, unsigned int  Flags)

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_memAlloc(JNIEnv* env, jobject obj, jint byteSize) {

	CUdeviceptr cuDevicePtr;

	CUresult cuResult = cuMemAlloc(&cuDevicePtr, (size_t)byteSize);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuDevicePtr;
}

//CUresult cuMemAllocHost(void** pp, size_t bytesize)

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_memAllocManaged(JNIEnv* env, jobject obj, jint byteSize, jint flags) {

	CUdeviceptr cuDevicePtr;

	CUresult cuResult = cuMemAllocManaged(&cuDevicePtr, (size_t)byteSize, (unsigned int)flags);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuDevicePtr;
}

//CUresult cuMemAllocPitch(CUdeviceptr * dptr, size_t * pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memFree(JNIEnv* env, jobject obj, jlong dptr) {

	CUresult cuResult = cuMemFree(dptr);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memFreeHost(JNIEnv* env, jobject obj, jlong p) {
	
	void* hostMemoryPointer = (void*)p;

	CUresult cuResult = cuMemFreeHost(hostMemoryPointer);

	return cuResult;
}

//CUresult cuMemGetAddressRange(CUdeviceptr * pbase, size_t * psize, CUdeviceptr dptr)
//CUresult cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags)
//CUresult cuMemGetInfo(size_t * free, size_t * total)
//CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int  Flags)
//CUresult cuMemHostGetDevicePointer(CUdeviceptr * pdptr, void* p, unsigned int  Flags)
//CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p)
//CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int  Flags)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memHostUnregister(JNIEnv* env, jobject obj, jlong p) {

	void* hostMemoryPointer = (void*)p;

	CUresult cuResult = cuMemHostUnregister(hostMemoryPointer);

	return cuResult;
}

//CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
//CUresult cuMemcpy2D(const CUDA_MEMCPY2D * pCopy)
//CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D * pCopy, CUstream hStream)
//CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D * pCopy)
//CUresult cuMemcpy3D(const CUDA_MEMCPY3D * pCopy)
//CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D * pCopy, CUstream hStream)
//CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER * pCopy)
//CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER * pCopy, CUstream hStream)
//CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
//CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount)
//CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount)
//CUresult cuMemcpyAtoH(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount)
//CUresult cuMemcpyAtoHAsync(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream)
//CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount)
//CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
//CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
//CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
//CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
//CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount)
//CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream)
//CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
//CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream)
//CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount)
//CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream)
//CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N)
//CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
//CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
//CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
//CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height)
//CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream)
//CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height)
//CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream)
//CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int  ui, size_t N)
//CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream)
//CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char  uc, size_t N)
//CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream)
//CUresult cuMipmappedArrayCreate(CUmipmappedArray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned int  numMipmapLevels)
//CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray)
//CUresult cuMipmappedArrayGetLevel(CUarray * pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int  level)
//CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUmipmappedArray mipmap, CUdevice device)
//CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUmipmappedArray mipmap)

//14. Virtual Memory Management //

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memAddressFree(JNIEnv* env, jobject obj, jlong ptr, jint size) {

	CUresult cuResult = cuMemAddressFree(ptr, (size_t)size);

	return cuResult;
}

//CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags)
//CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags)
//CUresult cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags)
//CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr)
//CUresult cuMemGetAllocationGranularity(size_t * granularity, const CUmemAllocationProp * prop, CUmemAllocationGranularity_flags option)
//CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp * prop, CUmemGenericAllocationHandle handle)
//CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle * handle, void* osHandle, CUmemAllocationHandleType shHandleType)
//CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags)
//CUresult cuMemMapArrayAsync(CUarrayMapInfo * mapInfoList, unsigned int  count, CUstream hStream)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memRelease(JNIEnv* env, jobject obj, jlong handle) {

	CUresult cuResult = cuMemRelease(handle);

	return cuResult;
}

//CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle * handle, void* addr)
//CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc * desc, size_t count)	

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memUnmap(JNIEnv* env, jobject obj, jlong ptr, jint size) {
	
	CUresult cuResult = cuMemUnmap(ptr, (size_t)size);

	return cuResult;
}


//15. Steam Ordered Memory Allocator //

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_memAllocAsync(JNIEnv* env, jobject obj, jint bytesize, jlong hStream) {

	CUdeviceptr cuDevicePtr;

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuMemAllocAsync(&cuDevicePtr, (size_t)bytesize, cuStream);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuDevicePtr;
}

//CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memFreeAsync(JNIEnv* env, jobject obj, jlong dptr, jlong hStream) {

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuMemFreeAsync(dptr, cuStream);

	return cuResult;
}

//CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memPoolDestroy(JNIEnv* env, jobject obj, jlong pool) {

	CUmemoryPool cuMemoryPool = reinterpret_cast<CUmemoryPool>(pool);

	CUresult cuResult = cuMemPoolDestroy(cuMemoryPool);

	return cuResult;
}

//CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr)
//CUresult cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags)
//CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location)
//CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value)
//CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags)
//CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData)
//CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count)
//CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memPoolTrimTo(JNIEnv* env, jobject obj, jlong pool, jint minBytesToKeep) {

	CUmemoryPool cuMemoryPool = reinterpret_cast<CUmemoryPool>(pool);

	CUresult cuResult = cuMemPoolTrimTo(cuMemoryPool, (size_t)minBytesToKeep);

	return cuResult;
}

//16. Multicast Object Management //

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_multicastAddDevice(JNIEnv* env, jobject obj, jlong mcHandle, jint dev) {

	CUresult cuResult = cuMulticastAddDevice(mcHandle, dev);

	return cuResult;
}

//CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags)
//CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags)
//CUresult cuMulticastCreate(CUmemGenericAllocationHandle* mcHandle, const CUmulticastObjectProp* prop)
//CUresult cuMulticastGetGranularity(size_t* granularity, const CUmulticastObjectProp* prop, CUmulticastGranularity_flags option)
//CUresult cuMulticastUnbind(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, size_t size)

//17. Unified Addressing
//CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device)
//CUresult cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location)
//CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream)
//CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int  flags, CUstream hStream)
//CUresult cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count)
//CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count)
//CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr)
//CUresult cuPointerGetAttributes(unsigned int  numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr)
//CUresult cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr)

//18. Stream Management //
//CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int  flags)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_streamAttachMemAsync(JNIEnv* env, jobject obj, jlong hStream, jlong dptr, jint length, jint flags) {
	
	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuStreamAttachMemAsync(cuStream, dptr, (size_t)length, (unsigned int)flags);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_streamBeginCapture(JNIEnv* env, jobject obj, jlong dst, jlong hStream, jint mode) {

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUstreamCaptureMode cuStreamCaptureMode =  static_cast<CUstreamCaptureMode>(mode);

	CUresult cuResult = cuStreamBeginCapture(cuStream, cuStreamCaptureMode);

	return cuResult;
}

//CUresult cuStreamBeginCaptureToGraph(CUstream hStream, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUstreamCaptureMode mode)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_streamCopyAttributes(JNIEnv* env, jobject obj, jlong dst, jlong src) {

	CUstream cuStreamDst = reinterpret_cast<CUstream>(dst);

	CUstream cuStreamSrc = reinterpret_cast<CUstream>(src);

	CUresult cuResult = cuStreamCopyAttributes(cuStreamDst, cuStreamSrc);

	return cuResult;
}


JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_streamCreate(JNIEnv* env, jobject obj, jint flags) {

	CUstream cuStream;

	CUresult cuResult = cuStreamCreate(&cuStream, (unsigned int)flags);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuStream;
}

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_streamCreateWithPriority(JNIEnv* env, jobject obj, jint flags, jint priority) {

	CUstream cuStream;

	CUresult cuResult = cuStreamCreateWithPriority(&cuStream, (unsigned int)flags, priority);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}
		
	return (jlong)cuStream;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_streamDestroy(JNIEnv* env, jobject obj, jlong hStream) {

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuStreamDestroy(cuStream);

	return cuResult;
}

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_streamEndCapture(JNIEnv* env, jobject obj, jlong hStream) {

	CUgraph cuGraph;

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuStreamEndCapture(cuStream, &cuGraph);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuGraph;
 }

//CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out)
//CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, size_t* numDependencies_out)
//CUresult cuStreamGetCaptureInfo_v3(CUstream hStream, CUstreamCaptureStatus * captureStatus_out, cuuint64_t * id_out, CUgraph * graph_out, const CUgraphNode * *dependencies_out, const CUgraphEdgeData * *edgeData_out, size_t * numDependencies_out)

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_streamGetCtx(JNIEnv* env, jobject obj, jlong hStream) {

	CUcontext pctx;

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuStreamGetCtx(cuStream, &pctx);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)pctx;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_streamGetFlags(JNIEnv* env, jobject obj, jlong hStream) {

	unsigned int flags;

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuStreamGetFlags(cuStream, &flags);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jint)flags;
}

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_streamGetId(JNIEnv* env, jobject obj, jlong hStream) {

	unsigned long long streamId;

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuStreamGetId(cuStream, &streamId);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)streamId;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_streamGetPriority(JNIEnv* env, jobject obj, jlong hStream) {
	
	int priority;

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuStreamGetPriority(cuStream, &priority);
	
	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return priority;
}

//CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus * captureStatus)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_streamQuery(JNIEnv* env, jobject obj, jlong hStream) {
	
	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuStreamQuery(cuStream);

	return cuResult;
}

//CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue * value)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_streamSynchronize(JNIEnv* env, jobject obj, jlong hStream) {

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuStreamSynchronize(cuStream);

	return cuResult;
}

//CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode * dependencies, size_t numDependencies, unsigned int  flags)
//CUresult cuStreamUpdateCaptureDependencies_v2(CUstream hStream, CUgraphNode * dependencies, const CUgraphEdgeData * dependencyData, size_t numDependencies, unsigned int  flags)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_streamWaitEvent(JNIEnv* env, jobject obj, jlong hStream, jlong hEvent, jint flags) {
	
	CUstream cuStream = reinterpret_cast<CUstream>(hStream);
	
	CUevent cuEvent = reinterpret_cast<CUevent>(hEvent);

	CUresult cuResult = cuStreamWaitEvent(cuStream, cuEvent, (unsigned int)flags);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_threadExchangeStreamCaptureMode(JNIEnv* env, jobject obj, jint mode) {
	
	CUstreamCaptureMode cuStreamCaptureMode = static_cast<CUstreamCaptureMode>(mode);

	CUresult cuResult = cuThreadExchangeStreamCaptureMode(&cuStreamCaptureMode);

	return cuResult;
}

//19. Event Management // COMPLETE
JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_eventCreate(JNIEnv* env, jobject obj, jint flags) {
	
	CUevent phEvent;

	CUresult cuResult = cuEventCreate(&phEvent, (unsigned int)flags);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)phEvent;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_eventDestroy(JNIEnv* env, jobject obj, jlong hEvent) {
	
	CUevent cuEvent = reinterpret_cast<CUevent>(hEvent);

	CUresult cuResult = cuEventDestroy(cuEvent);

	return cuResult;
}

JNIEXPORT jfloat JNICALL Java_kuda_driverapi_DriverAPI_eventElapsedTime(JNIEnv* env, jobject obj, jlong hStart, jlong hEnd) {

	float pMilliseconds;

	CUevent cuEventStart = reinterpret_cast<CUevent>(hStart);

	CUevent cuEventEnd = reinterpret_cast<CUevent>(hEnd);

	CUresult cuResult = cuEventElapsedTime(&pMilliseconds, cuEventStart, cuEventEnd);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return pMilliseconds;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_eventQuery(JNIEnv* env, jobject obj, jlong hEvent) {

	CUevent cuEvent = reinterpret_cast<CUevent>(hEvent);

	CUresult cuResult = cuEventQuery(cuEvent);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_eventRecord(JNIEnv* env, jobject obj, jlong hEvent, jlong hStream) {

	CUevent cuEvent = reinterpret_cast<CUevent>(hEvent);

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuEventRecord(cuEvent, cuStream);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_eventRecordWithFlags(JNIEnv* env, jobject obj, jlong hEvent, jlong hStream, jint flags) {

	CUevent cuEvent = reinterpret_cast<CUevent>(hEvent);

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuEventRecordWithFlags(cuEvent, cuStream, (unsigned int)flags);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_eventSynchronize(JNIEnv* env, jobject obj, jlong hEvent) {

	CUevent cuEvent = reinterpret_cast<CUevent>(hEvent);

	CUresult cuResult = cuEventSynchronize(cuEvent);

	return cuResult;
}

//20. External Resource Interoperability

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_destroyExternalMemory(JNIEnv* env, jobject obj, jlong extMem) {

	CUexternalMemory cuExternalMemory = reinterpret_cast<CUexternalMemory>(extMem);

	CUresult cuResult = cuDestroyExternalMemory(cuExternalMemory);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_destroyExternalSemaphore(JNIEnv* env, jobject obj, jlong extSem) {

	CUexternalSemaphore cuExternalSemaphore = reinterpret_cast<CUexternalSemaphore>(extSem);

	CUresult cuResult = cuDestroyExternalSemaphore(cuExternalSemaphore);

	return cuResult;
}

//CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc)
//CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc)
//CUresult cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc)
//CUresult cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc)
//CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream)
//CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream)

//21. Stream Memory Operations
//CUresult cuStreamBatchMemOp(CUstream stream, unsigned int  count, CUstreamBatchMemOpParams* paramArray, unsigned int  flags)
//CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags)
//CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags)
//CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags)
//CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags)

//22. Execution Control
//CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc)
//CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc)
//CUresult cuFuncGetName(const char** name, CUfunction hfunc)
//CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int  value)
//CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
//CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config)
//CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams)
//CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS * launchParamsList, unsigned int  numDevices, unsigned int  flags)
//CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData)
//CUresult cuLaunchKernel(CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
//CUresult cuLaunchKernelEx(const CUlaunchConfig * config, CUfunction f, void** kernelParams, void** extra)

//24. Graph Management
//CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_deviceGraphMemTrim(JNIEnv* env, jobject obj, jint device) {

	CUresult cuResult = cuDeviceGraphMemTrim(device);

	return cuResult;
}

//CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value)
//CUresult cuGraphAddBatchMemOpNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
//CUresult cuGraphAddChildGraphNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, CUgraph childGraph)
//CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t numDependencies)
//CUresult cuGraphAddDependencies_v2(CUgraph hGraph, const CUgraphNode * from, const CUgraphNode * to, const CUgraphEdgeData * edgeData, size_t numDependencies)
//CUresult cuGraphAddEmptyNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies)
//CUresult cuGraphAddEventRecordNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, CUevent event)
//CUresult cuGraphAddEventWaitNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, CUevent event)
//CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
//CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
//CUresult cuGraphAddHostNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS * nodeParams)
//CUresult cuGraphAddKernelNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
//CUresult cuGraphAddMemAllocNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS * nodeParams)
//CUresult cuGraphAddMemFreeNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, CUdeviceptr dptr)
//CUresult cuGraphAddMemcpyNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_MEMCPY3D * copyParams, CUcontext ctx)
//CUresult cuGraphAddMemsetNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext ctx)
//CUresult cuGraphAddNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, CUgraphNodeParams * nodeParams)
//CUresult cuGraphAddNode_v2(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, const CUgraphEdgeData * dependencyData, size_t numDependencies, CUgraphNodeParams * nodeParams)
//CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams_out)
//CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
//CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph * phGraph)
//CUresult cuGraphClone(CUgraph * phGraphClone, CUgraph originalGraph)
//CUresult cuGraphConditionalHandleCreate(CUgraphConditionalHandle * pHandle_out, CUgraph hGraph, CUcontext ctx, unsigned int  defaultLaunchValue, unsigned int  flags)

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_graphCreate(JNIEnv* env, jobject obj, jint flags) {
	
	CUgraph cuGraph;

	CUresult cuResult = cuGraphCreate(&cuGraph, (unsigned int)flags);

	return cuResult;
}

//CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int  flags)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_graphDestroy(JNIEnv* env, jobject obj, jlong hGraph) {

	CUgraph cuGraph= reinterpret_cast<CUgraph>(hGraph);

	CUresult cuResult = cuGraphDestroy(cuGraph);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_graphDestroyNode(JNIEnv* env, jobject obj, jlong hNode) {

	CUgraphNode cuGraphNode = reinterpret_cast<CUgraphNode>(hNode);

	CUresult cuResult = cuGraphDestroyNode(cuGraphNode);

	return cuResult;
}

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_graphEventRecordNodeGetEvent(JNIEnv* env, jobject obj, jlong hNode) {

	CUevent cuEvent;

	CUgraphNode cuGraphNode = reinterpret_cast<CUgraphNode>(hNode);

	CUresult cuResult = cuGraphEventRecordNodeGetEvent(cuGraphNode, &cuEvent);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuEvent;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_graphEventRecordNodeSetEvent(JNIEnv* env, jobject obj, jlong hNode, jlong event) {

	CUgraphNode cuGraphNode = reinterpret_cast<CUgraphNode>(hNode);

	CUevent cuEvent = reinterpret_cast<CUevent>(event);

	CUresult cuResult = cuGraphEventRecordNodeSetEvent(cuGraphNode, cuEvent);

	return cuResult;
}

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_graphEventWaitNodeGetEvent(JNIEnv* env, jobject obj, jlong hNode) {

	CUevent cuEvent;

	CUgraphNode cuGraphNode = reinterpret_cast<CUgraphNode>(hNode);

	CUresult cuResult = cuGraphEventWaitNodeGetEvent(cuGraphNode, &cuEvent);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuEvent;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_graphEventWaitNodeSetEvent(JNIEnv* env, jobject obj, jlong hNode, jlong event) {

	CUgraphNode cuGraphNode = reinterpret_cast<CUgraphNode>(hNode);

	CUevent cuEvent = reinterpret_cast<CUevent>(event);

	CUresult cuResult = cuGraphEventWaitNodeSetEvent(cuGraphNode, cuEvent);

	return cuResult;
}

//CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
//CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_graphExecDestroy(JNIEnv* env, jobject obj, jlong hGraphExec) {

	CUgraphExec cuGraphExec = reinterpret_cast<CUgraphExec>(hGraphExec);

	CUresult cuResult = cuGraphExecDestroy(cuGraphExec);

	return cuResult;
}

//CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
//CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
//CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
//CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
//CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t * flags)
//CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS * nodeParams)
//CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
//CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D * copyParams, CUcontext ctx)
//CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext ctx)
//CUresult cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraphNodeParams * nodeParams)
//CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo * resultInfo)
//CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * params_out)
//CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
//CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS * params_out)
//CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
//CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode * from, CUgraphNode * to, size_t * numEdges)
//CUresult cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode * from, CUgraphNode * to, CUgraphEdgeData * edgeData, size_t * numEdges)

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_graphGetNodes(JNIEnv* env, jobject obj, jlong hGraph) {

	CUgraph cuGraph = reinterpret_cast<CUgraph>(hGraph);

	CUgraphNode cuNodes;

	size_t numNodes;

	CUresult cuResult = cuGraphGetNodes(cuGraph, &cuNodes, &numNodes);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuNodes;
}

//CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode * rootNodes, size_t * numRootNodes)
//CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS * nodeParams)
//CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS * nodeParams)
//CUresult cuGraphInstantiate(CUgraphExec * phGraphExec, CUgraph hGraph, unsigned long long flags)
//CUresult cuGraphInstantiateWithParams(CUgraphExec * phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS * instantiateParams)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_graphKernelNodeCopyAttributes(JNIEnv* env, jobject obj, jlong dst, jlong src) {

	CUgraphNode cuGraphNodeDst = reinterpret_cast<CUgraphNode>(dst);

	CUgraphNode cuGraphNodeSrc = reinterpret_cast<CUgraphNode>(src);

	CUresult cuResult = cuGraphKernelNodeCopyAttributes(cuGraphNodeDst, cuGraphNodeSrc);

	return cuResult;
}

//CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue * value_out)
//CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS * nodeParams)
//CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue * value)
//CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_graphLaunch(JNIEnv* env, jobject obj, jlong hGraphExec, jlong hStream) {

	CUgraphExec cuGraphExec = reinterpret_cast<CUgraphExec>(hGraphExec);

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuGraphLaunch(cuGraphExec, cuStream);

	return cuResult;
}

//CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS * params_out)

JNIEXPORT jlong JNICALL Java_kuda_driverapi_DriverAPI_graphMemFreeNodeGetParams(JNIEnv* env, jobject obj, jlong hNode) {
	
	CUdeviceptr cuDevicePtr;

	CUgraphNode cuGraphNode = reinterpret_cast<CUgraphNode>(hNode);

	CUresult cuResult = cuGraphMemFreeNodeGetParams(cuGraphNode, &cuDevicePtr);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return (jlong)cuDevicePtr;
}

//CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D * nodeParams)
//CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D * nodeParams)
//CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS * nodeParams)
//CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS * nodeParams)
//CUresult cuGraphNodeFindInClone(CUgraphNode * phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph)
//CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode * dependencies, size_t * numDependencies)
//CUresult cuGraphNodeGetDependencies_v2(CUgraphNode hNode, CUgraphNode * dependencies, CUgraphEdgeData * edgeData, size_t * numDependencies)
//CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode * dependentNodes, size_t * numDependentNodes)
//CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode hNode, CUgraphNode * dependentNodes, CUgraphEdgeData * edgeData, size_t * numDependentNodes)
//CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled)
//CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType * type)
//CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int  isEnabled)
//CUresult cuGraphNodeSetParams(CUgraphNode hNode, CUgraphNodeParams * nodeParams)
//CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int  count)
//CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t numDependencies)
//CUresult cuGraphRemoveDependencies_v2(CUgraph hGraph, const CUgraphNode * from, const CUgraphNode * to, const CUgraphEdgeData * edgeData, size_t numDependencies)
//CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int  count, unsigned int  flags)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_graphUpload(JNIEnv* env, jobject obj, jlong hGraphExec, jlong hStream) {

	CUgraphExec cuGraphExec = reinterpret_cast<CUgraphExec>(hGraphExec);

	CUstream cuStream = reinterpret_cast<CUstream>(hStream);

	CUresult cuResult = cuGraphUpload(cuGraphExec, cuStream);

	return cuResult;
}

//CUresult cuUserObjectCreate(CUuserObject * object_out, void* ptr, CUhostFn destroy, unsigned int  initialRefcount, unsigned int  flags)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_userObjectRelease(JNIEnv* env, jobject obj, jlong object, jint count) {

	CUuserObject cuUserObject = reinterpret_cast<CUuserObject>(object);

	CUresult cuResult = cuUserObjectRelease(cuUserObject, (unsigned int) count);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_userObjectRetain(JNIEnv* env, jobject obj, jlong object, jint count) {

	CUuserObject cuObject = reinterpret_cast<CUuserObject>(object);

	CUresult cuResult = cuUserObjectRetain(cuObject, (unsigned int)count);

	return cuResult;
}

//25. Occupancy
//CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int  numBlocks, int  blockSize)
//CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize)
//CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags)
//CUresult cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config)
//CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit)
//CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit, unsigned int  flags)
//CUresult cuOccupancyMaxPotentialClusterSize(int* clusterSize, CUfunction func, const CUlaunchConfig * config)

//26. Texture Reference Management (DEPRECATED)

//27. Surface Reference Management (DEPRECATED)

//28. Texture Object Management
//CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_textObjectDestroy(JNIEnv* env, jobject obj, jlong textObject) {
	
	CUresult cuResult = cuTexObjectDestroy(textObject);

	return cuResult;
}

//CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject)
//CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC * pResViewDesc, CUtexObject texObject)
//CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC * pTexDesc, CUtexObject texObject)


//29. Surface Object Management
//CUresult cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_surfObjectDestroy(JNIEnv* env, jobject obj, jlong surfObject) {

	CUresult cuResult = cuSurfObjectDestroy(surfObject);

	return cuResult;
}

//CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject)

//30. Tensor Map Object Managment
//CUresult cuTensorMapEncodeIm2col(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const int* pixelBoxLowerCorner, const int* pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill)
//CUresult cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill)
//CUresult cuTensorMapReplaceAddress(CUtensorMap* tensorMap, void* globalAddress)

//31. Peer Context Memory Access

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxDisablePeerAccess(JNIEnv* env, jobject obj, jlong peerContext) {
	
	CUcontext cuContext = reinterpret_cast<CUcontext>(peerContext);
	
	CUresult cuResult = cuCtxDisablePeerAccess(cuContext);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ctxEnablePeerAccess(JNIEnv* env, jobject obj, jlong peerContext, jint flags) {

	CUcontext cuContext = reinterpret_cast<CUcontext>(peerContext);

	CUresult cuResult = cuCtxEnablePeerAccess(cuContext, (unsigned int)flags);

	return cuResult;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_deviceCanAccessPeer(JNIEnv* env, jobject obj, jint dev, jint peerDev) {

	int canAccessPeer;

	CUresult cuResult = cuDeviceCanAccessPeer(&canAccessPeer, dev, peerDev);

	if (cuResult != CUDA_SUCCESS) {
		return cuResult;
	}

	return canAccessPeer;
}

//CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice)

//32. Graphics Interoperability
//CUresult cuGraphicsMapResources(unsigned int  count, CUgraphicsResource* resources, CUstream hStream)
//CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray * pMipmappedArray, CUgraphicsResource resource)
//CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr * pDevPtr, size_t * pSize, CUgraphicsResource resource)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_graphicsResourceSetMapFlags(JNIEnv* env, jobject obj, jlong resource, jint flags) {

	CUgraphicsResource cuResource = reinterpret_cast<CUgraphicsResource>(resource);

	CUresult cuResult = cuGraphicsResourceSetMapFlags(cuResource, (unsigned int)flags);

	return cuResult;
}

//CUresult cuGraphicsSubResourceGetMappedArray(CUarray * pArray, CUgraphicsResource resource, unsigned int  arrayIndex, unsigned int  mipLevel)
//CUresult cuGraphicsUnmapResources(unsigned int  count, CUgraphicsResource * resources, CUstream hStream)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_graphicsUnregisterResource(JNIEnv* env, jobject obj, jlong resource) {

	CUgraphicsResource cuGraphicsResource = reinterpret_cast<CUgraphicsResource>(resource);

	CUresult cuResult = cuGraphicsUnregisterResource(cuGraphicsResource);

	return cuResult;
}

//33.Driver Entry Point Access
//CUresult cuGetProcAddress(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult * symbolStatus)

//34. Coredump Attributes Control API
//CUresult cuCoredumpGetAttribute(CUcoredumpSettings attrib, void* value, size_t * size)
//CUresult cuCoredumpGetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t * size)
//CUresult cuCoredumpSetAttribute(CUcoredumpSettings attrib, void* value, size_t * size)
//CUresult cuCoredumpSetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t * size)

//35. Profiler Control (DEPRECATED)

//36. Profiler Control

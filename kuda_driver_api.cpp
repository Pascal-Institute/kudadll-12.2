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

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_linkDestroy(JNIEnv* env, jobject obj, jlong state) {
	
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

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_libraryUnload(JNIEnv* env, jobject obj, jlong library) {
	
	CUlibrary cuLibrary = reinterpret_cast<CUlibrary>(library);

	CUresult cudaStatus = cuLibraryUnload(cuLibrary);

	return cudaStatus;
}

//13. Memory Management
//CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
//CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR * pArrayDescriptor, CUarray hArray)
//CUresult cuArrayCreate(CUarray * pHandle, const CUDA_ARRAY_DESCRIPTOR * pAllocateArray)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_arrayDestroy(JNIEnv* env, jobject obj, jlong hArray) {
	
	CUarray cuArray = reinterpret_cast<CUarray>(hArray);
	
	CUresult cudaStatus = cuArrayDestroy(cuArray);

	return cudaStatus;
}

//CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR * pArrayDescriptor, CUarray hArray)
//CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUarray array, CUdevice device)
//CUresult cuArrayGetPlane(CUarray * pPlaneArray, CUarray hArray, unsigned int  planeIdx)
//CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUarray array)
//CUresult cuDeviceGetByPCIBusId(CUdevice * dev, const char* pciBusId)
//CUresult cuDeviceGetPCIBusId(char* pciBusId, int  len, CUdevice dev)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_ipcCloseMemHandle(JNIEnv* env, jobject obj, jlong dptr) {

	CUresult cudaStatus = cuIpcCloseMemHandle(dptr);

	return cudaStatus;
}

//CUresult cuIpcGetEventHandle(CUipcEventHandle * pHandle, CUevent event)
//CUresult cuIpcGetMemHandle(CUipcMemHandle * pHandle, CUdeviceptr dptr)
//CUresult cuIpcOpenEventHandle(CUevent * phEvent, CUipcEventHandle handle)
//CUresult cuIpcOpenMemHandle(CUdeviceptr * pdptr, CUipcMemHandle handle, unsigned int  Flags)
//CUresult cuMemAlloc(CUdeviceptr * dptr, size_t bytesize)
//CUresult cuMemAllocHost(void** pp, size_t bytesize)
//CUresult cuMemAllocManaged(CUdeviceptr * dptr, size_t bytesize, unsigned int  flags)
//CUresult cuMemAllocPitch(CUdeviceptr * dptr, size_t * pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memFree(JNIEnv* env, jobject obj, jlong dptr) {

	CUresult cudaStatus = cuMemFree(dptr);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memFreeHost(JNIEnv* env, jobject obj, jlong p) {
	
	void* hostMemoryPointer = (void*)p;

	CUresult result = cuMemFreeHost(hostMemoryPointer);

	return result;
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

	CUresult result = cuMemHostUnregister(hostMemoryPointer);

	return result;
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

//14. Virtual Memory Management
//CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
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

	CUresult result = cuMemRelease(handle);

	return result;
}

//CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle * handle, void* addr)
//CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc * desc, size_t count)	
//CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)

//15. Steam Ordered Memory Allocator
//CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream)
//CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream)
//CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
//CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps)

JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_memPoolDestroy(JNIEnv* env, jobject obj, jlong pool) {

	CUmemoryPool cuMemoryPool = reinterpret_cast<CUmemoryPool>(pool);

	CUresult result = cuMemPoolDestroy(cuMemoryPool);

	return result;
}

//CUresult cuMemPoolDestroy(CUmemoryPool pool)
//CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr)
//CUresult cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags)
//CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location)
//CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value)
//CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags)
//CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData)
//CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count)
//CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value)
//CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep)
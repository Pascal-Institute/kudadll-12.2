#include <jni.h>

//https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

#ifdef __cplusplus
extern "C" {
#endif
	//6.1 Device Management
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_flushGPUDirectRDMAWrites(JNIEnv* env, jclass cls, jint scope);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_DeviceHandler_getDefaultMemPool(JNIEnv* env, jclass cls, jint  device);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getLimit(JNIEnv* env, jclass cls, jbyte limit);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_DeviceHandler_getMemPool(JNIEnv* env, jclass cls, jint  device);

	JNIEXPORT jstring JNICALL Java_kuda_runtimeapi_DeviceHandler_getPCIBusId(JNIEnv* env, jclass cls, jint device);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getStreamPriorityRange(JNIEnv* env, jclass cls);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setCacheConfig(JNIEnv* env, jclass cls, jint cacheConfig);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setLimit(JNIEnv* env, jclass cls, jbyte limit, jsize value);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setSharedMemConfig(JNIEnv* env, jclass cls, jint config);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_synchronize(JNIEnv* env, jclass cls);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_reset(JNIEnv* env, jclass cls);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_getDevice(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_getDiviceCount(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_initDevice(JNIEnv* env, jobject obj, jint device, jint deviceFlags, jint flags);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_lpcCloseMemHandle(JNIEnv* env, jobject obj, jlong devicePtr);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_setDevice(JNIEnv* env, jobject obj, jint device);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_setDeviceFlags(JNIEnv* env, jobject obj, jint flags);

	//6.3 Error Handling
	JNIEXPORT jstring JNICALL Java_kuda_runtimeapi_RuntimeAPI_getErrorName(JNIEnv* env, jobject obj, jint error);

	JNIEXPORT jstring JNICALL Java_kuda_runtimeapi_RuntimeAPI_getErrorString(JNIEnv* env, jobject obj, jint error);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_getLastError(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_peekAtLastError(JNIEnv* env, jobject obj);

	//6.4 Stream Management
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_ctxResetPersistingL2Cache(JNIEnv* env, jclass cls);

	//cudaStreamAddCallback

	//cudaStreamAttachMemAsync

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_beginCapture(JNIEnv* env, jclass cls, jlong stream, jint mode);

	//cudaStreamBeginCaptureToGraph

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_copyAttributes(JNIEnv* env, jclass cls, jlong dst, jlong src);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_StreamHandler_create(JNIEnv* env, jclass cls);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_StreamHandler_createWithFlags(JNIEnv* env, jclass cls, jint flags);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_StreamHandler_createWithPriority(JNIEnv* env, jclass cls, jint flags, jint priority);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_destory(JNIEnv* env, jclass cls, jlong stream);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_query(JNIEnv* env, jclass cls, jlong stream);

	//cudaStreamSetAttribute

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_synchrnoize(JNIEnv* env, jclass cls, jlong stream);

	//cudaStreamUpdateCaptureDependencies

	//cudaStreamUpdateCaptureDependencies_v2

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_waitEvent(JNIEnv* env, jclass cls, jlong stream, jlong event, jint flags);

	//cudaThreadExchangeStreamCaptureMode

	//6.5 Event ManageMent
	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_EventHandler_create(JNIEnv* env, jclass cls);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_EventHandler_createWithFlags(JNIEnv* env, jclass cls, jint flags);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_destroy(JNIEnv* env, jclass cls, jlong event);

	JNIEXPORT jfloat JNICALL Java_kuda_runtimeapi_EventHandler_elapsedTime(JNIEnv* env, jclass cls, jlong start, jlong end);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_query(JNIEnv* env, jclass cls, jlong event);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_record(JNIEnv* env, jclass cls, jlong event, jlong stream);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_recordWithFlags(JNIEnv* env, jclass cls, jlong event, jlong stream, jint flags);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_synchronize(JNIEnv* env, jclass cls, jlong event);

	//6.6 External Reource Interoperability
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_destroyExternalMemory(JNIEnv* env, jobject obj, jlong extMem);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_destroyExternalSemaphore(JNIEnv* env, jobject obj, jlong extSem);

	//JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_externalMemoryGetMappedBuffer(JNIEnv* env, jobject obj);
	
	//JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_externalMemoryGetMappedMipmappedArray(JNIEnv* env, jobject obj);
	
	//JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_importExternalMemory(JNIEnv* env, jobject obj);
		
	//JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_importExternalSemaphore(JNIEnv* env, jobject obj);
		
	//JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_signalExternalSemaphoresAsync(JNIEnv* env, jobject obj);
		
	//JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_waitExternalSemaphoresAsync(JNIEnv* env, jobject obj);

	//6.9 Memory Management
	//__host__​cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array)
	//__host__​cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int  device)
	//__host__​cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int  planeIdx)
	//__host__​cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array)
	
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_free(JNIEnv* env, jobject obj, jlong devPtr);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_freeArray(JNIEnv* env, jobject obj, jlong array);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_freeHost(JNIEnv* env, jobject obj, jlong ptr);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_freeMipmappedArray(JNIEnv* env, jobject obj, jlong mipMappedArray);

	//__host__​cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int  level)
	//__host__​cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol)
	//__host__​cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol)

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_RuntimeAPI_hostAlloc(JNIEnv* env, jobject obj, jsize size, jint flags);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_RuntimeAPI_hostRegister(JNIEnv* env, jobject obj, jsize size, jint flags);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_hostUnregister(JNIEnv* env, jobject obj, jlong ptr);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_RuntimeAPI_malloc(JNIEnv* env, jobject obj, jsize size);

	//__host__​cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent)
	//__host__​cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned int  flags = 0)
	//__host__​cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned int  flags = 0)

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_RuntimeAPI_mallocHost(JNIEnv* env, jobject obj, jsize size);

	//6.13  Peer Device Memory Access
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_deviceCanAccessPeer(JNIEnv* env, jobject obj, jint  device, jint  peerDevice);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_deviceDisablePeerAccess(JNIEnv* env, jobject obj, jint peerDevice);
	
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_deviceEnablePeerAccess(JNIEnv* env, jobject obj, jint  peerDevice, jint flags);

	//6.27 Version Management
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_driverGetVersion(JNIEnv* env, jobject obj);
	
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_runtimeGetVersion(JNIEnv* env, jobject obj);

#ifdef __cplusplus
}
#endif
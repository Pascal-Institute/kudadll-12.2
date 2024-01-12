#include "kuda_runtime_api.h"
#include <jni.h>
#include <string>
#include <cuda_runtime_api.h>

//6.1 Device Management
JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_chooseDevice(JNIEnv* env, jclass cls, jobject deviceProp) {

	cudaDeviceProp cDeviceProp;

	jclass devicePropClass = env->FindClass("kuda/runtimeapi/structure/DeviceProp");

	jfieldID fid;

	fid = env->GetFieldID(devicePropClass, "eccEnabled", "I");
	cDeviceProp.ECCEnabled = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "accessPolicyMaxWindowSize", "I");
	cDeviceProp.accessPolicyMaxWindowSize = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "asyncEngineCount", "I");
	cDeviceProp.asyncEngineCount = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "canMapHostMemory", "I");
	cDeviceProp.canMapHostMemory = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "canUseHostPointerForRegisteredMem", "I");
	cDeviceProp.canUseHostPointerForRegisteredMem = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "clockRate", "I");
	cDeviceProp.clockRate = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "clusterLaunch", "I");
	cDeviceProp.clusterLaunch = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "computeMode", "I");
	cDeviceProp.computeMode = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "computePreemptionSupported", "I");
	cDeviceProp.computePreemptionSupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "concurrentKernels", "I");
	cDeviceProp.concurrentKernels = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "concurrentManagedAccess", "I");
	cDeviceProp.concurrentManagedAccess = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "cooperativeLaunch", "I");
	cDeviceProp.cooperativeLaunch = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "cooperativeMultiDeviceLaunch", "I");
	cDeviceProp.cooperativeMultiDeviceLaunch = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "deferredMappingCudaArraySupported", "I");
	cDeviceProp.deferredMappingCudaArraySupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "deviceOverlap", "I");
	cDeviceProp.deviceOverlap = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "directManagedMemAccessFromHost", "I");
	cDeviceProp.directManagedMemAccessFromHost = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "globalL1CacheSupported", "I");
	cDeviceProp.globalL1CacheSupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "gpuDirectRDMAFlushWritesOptions", "I");
	cDeviceProp.gpuDirectRDMAFlushWritesOptions = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "gpuDirectRDMASupported", "I");
	cDeviceProp.gpuDirectRDMASupported = env->GetIntField(deviceProp, fid);
	
	fid = env->GetFieldID(devicePropClass, "gpuDirectRDMAWritesOrdering", "I");
	cDeviceProp.gpuDirectRDMAWritesOrdering = env->GetIntField(deviceProp, fid);
	
	fid = env->GetFieldID(devicePropClass, "hostNativeAtomicSupported", "I");
	cDeviceProp.hostNativeAtomicSupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "hostRegisterReadOnlySupported", "I");
	cDeviceProp.hostRegisterReadOnlySupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "hostRegisterSupported", "I");
	cDeviceProp.hostRegisterSupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "integrated", "I");
	cDeviceProp.integrated = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "ipcEventSupported", "I");
	cDeviceProp.ipcEventSupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "isMultiGpuBoard", "I");
	cDeviceProp.isMultiGpuBoard = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "kernelExecTimeoutEnabled", "I");
	cDeviceProp.kernelExecTimeoutEnabled = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "l2CacheSize", "I");
	cDeviceProp.l2CacheSize = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "localL1CacheSupported", "I");
	cDeviceProp.localL1CacheSupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "luid", "Ljava/lang/String;");
	jstring luidString = (jstring)env->GetObjectField(deviceProp, fid);
	const char* luidChars = env->GetStringUTFChars(luidString, nullptr);
	strcpy_s(cDeviceProp.luid, luidChars);
	env->ReleaseStringUTFChars(luidString, luidChars);

	fid = env->GetFieldID(devicePropClass, "luidDeviceNodeMask", "I");
	cDeviceProp.luidDeviceNodeMask = env->GetIntField(deviceProp, fid);
	
	fid = env->GetFieldID(devicePropClass, "major", "I");
	cDeviceProp.major = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "managedMemory", "I");
	cDeviceProp.managedMemory = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "maxBlocksPerMultiProcessor", "I");
	cDeviceProp.maxBlocksPerMultiProcessor = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "maxGridSize", "[I");
	jintArray maxGridSizeArray = (jintArray) env->GetObjectField(deviceProp, fid);
	jint* maxGridSizeArrayElements =  env->GetIntArrayElements(maxGridSizeArray, nullptr);
	std::copy(maxGridSizeArrayElements, maxGridSizeArrayElements + 3, cDeviceProp.maxGridSize);
	env->ReleaseIntArrayElements(maxGridSizeArray, maxGridSizeArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxSurface1D", "I");
	cDeviceProp.maxSurface1D = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "maxSurface1DLayered", "[I");
	jintArray maxSurface1DLayeredArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxSurface1DLayeredArrayElements = env->GetIntArrayElements(maxSurface1DLayeredArray, nullptr);
	std::copy(maxSurface1DLayeredArrayElements, maxSurface1DLayeredArrayElements + 2, cDeviceProp.maxSurface1DLayered);
	env->ReleaseIntArrayElements(maxSurface1DLayeredArray, maxSurface1DLayeredArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxSurface2DLayered", "[I");
	jintArray maxSurface2DLayeredArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxSurface2DLayeredArrayElements = env->GetIntArrayElements(maxSurface2DLayeredArray, nullptr);
	std::copy(maxSurface2DLayeredArrayElements, maxSurface2DLayeredArrayElements + 3, cDeviceProp.maxSurface2DLayered);
	env->ReleaseIntArrayElements(maxSurface2DLayeredArray, maxSurface2DLayeredArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxSurface3D", "[I");
	jintArray maxSurface3DArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxSurface3DArrayElements = env->GetIntArrayElements(maxSurface3DArray, nullptr);
	std::copy(maxSurface3DArrayElements, maxSurface3DArrayElements + 3, cDeviceProp.maxSurface3D);
	env->ReleaseIntArrayElements(maxSurface3DArray, maxSurface3DArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxSurfaceCubemap", "I");
	cDeviceProp.maxSurfaceCubemap = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "maxSurfaceCubemapLayered", "[I");
	jintArray maxSurfaceCubemapLayeredArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxSurfaceCubemapLayeredArrayElements = env->GetIntArrayElements(maxSurfaceCubemapLayeredArray, nullptr);
	std::copy(maxSurfaceCubemapLayeredArrayElements, maxSurfaceCubemapLayeredArrayElements + 2, cDeviceProp.maxSurfaceCubemapLayered);
	env->ReleaseIntArrayElements(maxSurfaceCubemapLayeredArray, maxSurfaceCubemapLayeredArrayElements, JNI_ABORT);
	
	fid = env->GetFieldID(devicePropClass, "maxTexture1D", "I");
	cDeviceProp.maxTexture1D = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "maxTexture1DLayered", "[I");
	jintArray maxTexture1DLayeredArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxTexture1DLayeredArrayElements = env->GetIntArrayElements(maxTexture1DLayeredArray, nullptr);
	std::copy(maxTexture1DLayeredArrayElements, maxTexture1DLayeredArrayElements + 2, cDeviceProp.maxTexture1DLayered);
	env->ReleaseIntArrayElements(maxTexture1DLayeredArray, maxTexture1DLayeredArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxTexture1DLinear", "I");
	cDeviceProp.maxTexture1DLinear = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "maxTexture1DMipmap", "I");
	cDeviceProp.maxTexture1DMipmap = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "maxSurface2D", "[I");
	jintArray maxSurface2DArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxSurface2DArrayElements = env->GetIntArrayElements(maxSurface2DArray, nullptr);
	std::copy(maxSurface2DArrayElements, maxSurface2DArrayElements + 2, cDeviceProp.maxSurface2D);
	env->ReleaseIntArrayElements(maxSurface2DArray, maxSurface2DArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxTexture2DGather", "[I");
	jintArray maxTexture2DGatherArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxTexture2DGatherArrayElements = env->GetIntArrayElements(maxTexture2DGatherArray, nullptr);
	std::copy(maxTexture2DGatherArrayElements, maxTexture2DGatherArrayElements + 2, cDeviceProp.maxTexture2DGather);
	env->ReleaseIntArrayElements(maxTexture2DGatherArray, maxTexture2DGatherArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxTexture2DLayered", "[I");
	jintArray maxTexture2DLayeredArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxTexture2DLayeredArrayElements = env->GetIntArrayElements(maxTexture2DLayeredArray, nullptr);
	std::copy(maxTexture2DLayeredArrayElements, maxTexture2DLayeredArrayElements + 3, cDeviceProp.maxTexture2DLayered);
	env->ReleaseIntArrayElements(maxTexture2DLayeredArray, maxTexture2DLayeredArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxTexture2DLinear", "[I");
	jintArray maxTexture2DLinearArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxTexture2DLinearArrayElements = env->GetIntArrayElements(maxTexture2DLinearArray, nullptr);
	std::copy(maxTexture2DLinearArrayElements, maxTexture2DLinearArrayElements + 3, cDeviceProp.maxTexture2DLinear);
	env->ReleaseIntArrayElements(maxTexture2DLinearArray, maxTexture2DLinearArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxTexture2DMipmap", "[I");
	jintArray maxTexture2DMipmapArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxTexture2DMipmapArrayElements = env->GetIntArrayElements(maxTexture2DMipmapArray, nullptr);
	std::copy(maxTexture2DMipmapArrayElements, maxTexture2DMipmapArrayElements + 2, cDeviceProp.maxTexture2DMipmap);
	env->ReleaseIntArrayElements(maxTexture2DMipmapArray, maxTexture2DMipmapArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxTexture3D", "[I");
	jintArray maxTexture3DArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxTexture3DArrayElements = env->GetIntArrayElements(maxTexture3DArray, nullptr);
	std::copy(maxTexture3DArrayElements, maxTexture3DArrayElements + 3, cDeviceProp.maxTexture3D);
	env->ReleaseIntArrayElements(maxTexture3DArray, maxTexture3DArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxTexture3DAlt", "[I");
	jintArray maxSurface3DAltArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxSurface3DAltArrayElements = env->GetIntArrayElements(maxSurface3DAltArray, nullptr);
	std::copy(maxSurface3DAltArrayElements, maxSurface3DAltArrayElements + 3, cDeviceProp.maxTexture3DAlt);
	env->ReleaseIntArrayElements(maxSurface3DAltArray, maxSurface3DAltArrayElements, JNI_ABORT);
	
	fid = env->GetFieldID(devicePropClass, "maxTextureCubemap", "I");
	cDeviceProp.maxTextureCubemap = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "maxTextureCubemapLayered", "[I");
	jintArray maxTextureCubemapLayeredArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxTextureCubemapLayeredArrayElements = env->GetIntArrayElements(maxTextureCubemapLayeredArray, nullptr);
	std::copy(maxTextureCubemapLayeredArrayElements, maxTextureCubemapLayeredArrayElements + 2, cDeviceProp.maxTextureCubemapLayered);
	env->ReleaseIntArrayElements(maxTextureCubemapLayeredArray, maxTextureCubemapLayeredArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxThreadsDim", "[I");
	jintArray maxThreadsDimdArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* maxThreadsDimdArrayElements = env->GetIntArrayElements(maxThreadsDimdArray, nullptr);
	std::copy(maxThreadsDimdArrayElements, maxThreadsDimdArrayElements + 3, cDeviceProp.maxThreadsDim);
	env->ReleaseIntArrayElements(maxThreadsDimdArray, maxThreadsDimdArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "maxThreadsPerBlock", "I");
	cDeviceProp.maxThreadsPerBlock = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "maxThreadsPerMultiProcessor", "I");
	cDeviceProp.maxThreadsPerMultiProcessor = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "memPitch", "J");
	cDeviceProp.memPitch = (size_t)env->GetLongField(deviceProp, fid);
	
	fid = env->GetFieldID(devicePropClass, "memoryBusWidth", "I");
	cDeviceProp.memoryBusWidth = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "memoryClockRate", "I");
	cDeviceProp.memoryClockRate = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "memoryPoolSupportedHandleTypes", "I");
	cDeviceProp.memoryPoolSupportedHandleTypes = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "memoryPoolsSupported", "I");
	cDeviceProp.memoryPoolsSupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "minor", "I");
	cDeviceProp.minor = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "multiGpuBoardGroupID", "I");
	cDeviceProp.multiGpuBoardGroupID = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "multiProcessorCount", "I");
	cDeviceProp.multiProcessorCount = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "name", "Ljava/lang/String;");
	jstring nameString = (jstring)env->GetObjectField(deviceProp, fid);
	const char* nameChars = env->GetStringUTFChars(nameString, nullptr);
	strcpy_s(cDeviceProp.name, nameChars);
	env->ReleaseStringUTFChars(nameString, nameChars);

	fid = env->GetFieldID(devicePropClass, "pageableMemoryAccess", "I");
	cDeviceProp.pageableMemoryAccess = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "pageableMemoryAccessUsesHostPageTables", "I");
	cDeviceProp.pageableMemoryAccessUsesHostPageTables = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "pciBusID", "I");
	cDeviceProp.pciBusID = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "pciDeviceID", "I");
	cDeviceProp.pciDeviceID = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "pciDomainID", "I");
	cDeviceProp.pciDomainID = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "persistingL2CacheMaxSize", "I");
	cDeviceProp.persistingL2CacheMaxSize = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "regsPerBlock", "I");
	cDeviceProp.regsPerBlock = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "regsPerMultiprocessor", "I");
	cDeviceProp.regsPerMultiprocessor = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "reserved", "[I");
	jintArray reservedArray = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* reservedArrayElements = env->GetIntArrayElements(reservedArray, nullptr);
	std::copy(reservedArrayElements, reservedArrayElements + 61, cDeviceProp.reserved);
	env->ReleaseIntArrayElements(reservedArray, reservedArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "reserved2", "[I");
	jintArray reserved2Array = (jintArray)env->GetObjectField(deviceProp, fid);
	jint* reserved2ArrayElements = env->GetIntArrayElements(reserved2Array, nullptr);
	std::copy(reserved2ArrayElements, reserved2ArrayElements + 2, cDeviceProp.reserved2);
	env->ReleaseIntArrayElements(reserved2Array, reserved2ArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "reservedSharedMemPerBlock", "J");
	cDeviceProp.reservedSharedMemPerBlock = env->GetIntField(deviceProp, fid);
	
	fid = env->GetFieldID(devicePropClass, "sharedMemPerBlock", "J");
	cDeviceProp.sharedMemPerBlock = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "sharedMemPerBlockOptin", "J");
	cDeviceProp.sharedMemPerBlockOptin = env->GetIntField(deviceProp, fid);
	
	fid = env->GetFieldID(devicePropClass, "sharedMemPerMultiprocessor", "J");
	cDeviceProp.sharedMemPerMultiprocessor = env->GetIntField(deviceProp, fid);
	
	fid = env->GetFieldID(devicePropClass, "singleToDoublePrecisionPerfRatio", "I");
	cDeviceProp.singleToDoublePrecisionPerfRatio = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "sparseCudaArraySupported", "I");
	cDeviceProp.sparseCudaArraySupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "streamPrioritiesSupported", "I");
	cDeviceProp.streamPrioritiesSupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "surfaceAlignment", "J");
	cDeviceProp.surfaceAlignment = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "tccDriver", "I");
	cDeviceProp.tccDriver = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "textureAlignment", "J");
	cDeviceProp.textureAlignment = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "texturePitchAlignment", "J");
	cDeviceProp.texturePitchAlignment = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "timelineSemaphoreInteropSupported", "I");
	cDeviceProp.timelineSemaphoreInteropSupported = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "totalConstMem", "J");
	cDeviceProp.totalConstMem = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "totalGlobalMem", "J");
	cDeviceProp.totalGlobalMem = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "unifiedAddressing", "I");
	cDeviceProp.unifiedAddressing = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "unifiedFunctionPointers", "I");
	cDeviceProp.unifiedFunctionPointers = env->GetIntField(deviceProp, fid);

	fid = env->GetFieldID(devicePropClass, "uuid", "[B");
	jbyteArray uuidArray = (jbyteArray)env->GetObjectField(deviceProp, fid);
	jbyte* uuidArrayElements = env->GetByteArrayElements(uuidArray, nullptr);
	for (int i = 0; i < 16; ++i) {
		cDeviceProp.uuid.bytes[i] = static_cast<char>(uuidArrayElements[i]);
	}
	env->ReleaseByteArrayElements(uuidArray, uuidArrayElements, JNI_ABORT);

	fid = env->GetFieldID(devicePropClass, "warpSize", "I");
	cDeviceProp.warpSize = env->GetIntField(deviceProp, fid);

	int device;
	
	cudaError_t cudaStatus = cudaChooseDevice(&device, &cDeviceProp);
	
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return (jint)device;
}


JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_flushGPUDirectRDMAWrites(JNIEnv* env, jclass cls, jint scope) {
	
	cudaFlushGPUDirectRDMAWritesTarget e = cudaFlushGPUDirectRDMAWritesTargetCurrentDevice;

	cudaError_t cudaStatus = cudaDeviceFlushGPUDirectRDMAWrites(e, static_cast<cudaFlushGPUDirectRDMAWritesScope>(scope));
	
	return cudaStatus;
}

JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_DeviceHandler_getDefaultMemPool(JNIEnv* env, jclass cls, jint  device) {

	cudaMemPool_t memPool;

	cudaError_t cudaStatus = cudaDeviceGetMemPool(&memPool, (int)device);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return (jlong)memPool;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getLimit(JNIEnv* env, jclass cls, jbyte limit) {

	size_t pValue;

	cudaError_t cudaStatus = cudaDeviceGetLimit(&pValue, static_cast<cudaLimit>(limit));

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return pValue;
}

JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_DeviceHandler_getMemPool(JNIEnv* env, jclass cls, jint  device) {

	cudaMemPool_t memPool;

	cudaError_t cudaStatus = cudaDeviceGetMemPool(&memPool, (int)device);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return (jlong)memPool;
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

	cudaError_t cudaStatus = cudaDeviceSetLimit(static_cast<cudaLimit>(limit), (size_t)value);

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

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setValidDevices(JNIEnv* env, jclass cls, jint len) {

	int device_arr;

	cudaError_t cudaStatus = cudaSetValidDevices(&device_arr, len);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return device_arr;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getDevice(JNIEnv* env, jobject instance) {

	int diviceCode;

	cudaError_t cudaStatus = cudaGetDevice(&diviceCode);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return diviceCode;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getDiviceCount(JNIEnv* env, jobject instance) {
	int diviceCount;

	cudaError_t cudaStatus = cudaGetDeviceCount(&diviceCount);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return diviceCount;
}

JNIEXPORT jobject JNICALL Java_kuda_runtimeapi_DeviceHandler_getDeviceProperties(JNIEnv* env, jobject obj, jint device) {
	cudaDeviceProp cudaDeviceProp;

	cudaError_t cudaStatus = cudaGetDeviceProperties(&cudaDeviceProp, device);

	if (cudaStatus != cudaSuccess) {
		return nullptr;
	}

	jintArray maxGridSizeArray = env->NewIntArray(3);
	env->SetIntArrayRegion(maxGridSizeArray, 0, 3, reinterpret_cast<const jint*>(cudaDeviceProp.maxGridSize));

	jintArray maxSurface1DLayeredArray = env->NewIntArray(2);
	env->SetIntArrayRegion(maxSurface1DLayeredArray, 0, 2, reinterpret_cast<const jint*>(cudaDeviceProp.maxSurface1DLayered));

	jintArray maxSurface2DArray = env->NewIntArray(2);
	env->SetIntArrayRegion(maxSurface2DArray, 0, 2, reinterpret_cast<const jint*>(cudaDeviceProp.maxSurface2D));

	jintArray maxSurface2DLayeredArray = env->NewIntArray(3);
	env->SetIntArrayRegion(maxSurface2DLayeredArray, 0, 3, reinterpret_cast<const jint*>(cudaDeviceProp.maxSurface2DLayered));

	jintArray maxSurface3DArray = env->NewIntArray(3);
	env->SetIntArrayRegion(maxSurface3DArray, 0, 3, reinterpret_cast<const jint*>(cudaDeviceProp.maxSurface3D));

	jintArray maxSurfaceCubemapLayeredArray = env->NewIntArray(2);
	env->SetIntArrayRegion(maxSurfaceCubemapLayeredArray, 0, 2, reinterpret_cast<const jint*>(cudaDeviceProp.maxSurfaceCubemapLayered));

	jintArray maxTexture1DLayeredArray = env->NewIntArray(2);
	env->SetIntArrayRegion(maxTexture1DLayeredArray, 0, 2, reinterpret_cast<const jint*>(cudaDeviceProp.maxTexture1DLayered));

	jintArray maxTexture2DArray = env->NewIntArray(2);
	env->SetIntArrayRegion(maxTexture2DArray, 0, 2, reinterpret_cast<const jint*>(cudaDeviceProp.maxTexture2D));

	jintArray maxTexture2DGatherArray = env->NewIntArray(2);
	env->SetIntArrayRegion(maxTexture2DGatherArray, 0, 2, reinterpret_cast<const jint*>(cudaDeviceProp.maxTexture2DGather));

	jintArray maxTexture2DLayeredArray = env->NewIntArray(3);
	env->SetIntArrayRegion(maxTexture2DLayeredArray, 0, 3, reinterpret_cast<const jint*>(cudaDeviceProp.maxTexture2DLayered));

	jintArray maxTexture2DLinearArray = env->NewIntArray(3);
	env->SetIntArrayRegion(maxTexture2DLinearArray, 0, 3, reinterpret_cast<const jint*>(cudaDeviceProp.maxTexture2DLinear));

	jintArray maxTexture2DMipmapArray = env->NewIntArray(2);
	env->SetIntArrayRegion(maxTexture2DMipmapArray, 0, 2, reinterpret_cast<const jint*>(cudaDeviceProp.maxTexture2DMipmap));

	jintArray maxTexture3DArray = env->NewIntArray(3);
	env->SetIntArrayRegion(maxTexture3DArray, 0, 3, reinterpret_cast<const jint*>(cudaDeviceProp.maxTexture3D));

	jintArray maxTexture3DAltArray = env->NewIntArray(3);
	env->SetIntArrayRegion(maxTexture3DAltArray, 0, 3, reinterpret_cast<const jint*>(cudaDeviceProp.maxTexture3DAlt));

	jintArray maxTextureCubemapLayeredArray = env->NewIntArray(2);
	env->SetIntArrayRegion(maxTextureCubemapLayeredArray, 0, 2, reinterpret_cast<const jint*>(cudaDeviceProp.maxTextureCubemapLayered));

	jintArray maxThreadsDimArray = env->NewIntArray(3);
	env->SetIntArrayRegion(maxThreadsDimArray, 0, 3, reinterpret_cast<const jint*>(cudaDeviceProp.maxThreadsDim));

	jintArray reservedArray = env->NewIntArray(61);
	env->SetIntArrayRegion(reservedArray, 0, 61, reinterpret_cast<const jint*>(cudaDeviceProp.reserved));

	jintArray reserved2Array = env->NewIntArray(2);
	env->SetIntArrayRegion(reserved2Array, 0, 2, reinterpret_cast<const jint*>(cudaDeviceProp.reserved2));

	jbyteArray uuidArray = env->NewByteArray(16);
	env->SetByteArrayRegion(uuidArray, 0, 16, reinterpret_cast<const jbyte*>(cudaDeviceProp.uuid.bytes));

	jclass cudaDevicePropertiesClass = env->FindClass("kuda/runtimeapi/structure/DeviceProp");

	jmethodID constructor = env->GetMethodID(cudaDevicePropertiesClass, "<init>", "(IIIIIIIIIIIIIIIIIIIIIIIIIIIIILjava/lang/String;IIII[II[I[I[I[II[II[III[I[I[I[I[I[I[II[I[IIIJIIIIIIILjava/lang/String;IIIIIIII[I[IJJJJIIIJIJJIJJII[BI)V");
	jobject cudaDevicePropertiesObject = env->NewObject(cudaDevicePropertiesClass, constructor,
		cudaDeviceProp.ECCEnabled,
		cudaDeviceProp.accessPolicyMaxWindowSize,
		cudaDeviceProp.asyncEngineCount,
		cudaDeviceProp.canMapHostMemory,
		cudaDeviceProp.canUseHostPointerForRegisteredMem,
		cudaDeviceProp.clockRate,
		cudaDeviceProp.clusterLaunch,
		cudaDeviceProp.computeMode,
		cudaDeviceProp.computePreemptionSupported,
		cudaDeviceProp.concurrentKernels,

		cudaDeviceProp.concurrentManagedAccess,
		cudaDeviceProp.cooperativeLaunch,
		cudaDeviceProp.cooperativeMultiDeviceLaunch,
		cudaDeviceProp.deferredMappingCudaArraySupported,
		cudaDeviceProp.deviceOverlap,
		cudaDeviceProp.directManagedMemAccessFromHost,
		cudaDeviceProp.globalL1CacheSupported,
		cudaDeviceProp.gpuDirectRDMAFlushWritesOptions,
		cudaDeviceProp.gpuDirectRDMASupported,
		cudaDeviceProp.gpuDirectRDMAWritesOrdering,

		cudaDeviceProp.hostNativeAtomicSupported,
		cudaDeviceProp.hostRegisterReadOnlySupported,
		cudaDeviceProp.hostRegisterSupported,
		cudaDeviceProp.integrated,
		cudaDeviceProp.ipcEventSupported,
		cudaDeviceProp.isMultiGpuBoard,
		cudaDeviceProp.kernelExecTimeoutEnabled,
		cudaDeviceProp.l2CacheSize,
		cudaDeviceProp.localL1CacheSupported,
		env->NewStringUTF(cudaDeviceProp.luid),
		
		cudaDeviceProp.luidDeviceNodeMask,
		cudaDeviceProp.major,
		cudaDeviceProp.managedMemory,
		cudaDeviceProp.maxBlocksPerMultiProcessor,
		maxGridSizeArray,
		cudaDeviceProp.maxSurface1D,
		maxSurface1DLayeredArray,
		maxSurface2DArray,
		maxSurface2DLayeredArray,
		maxSurface3DArray,
		
		cudaDeviceProp.maxSurfaceCubemap,
		maxSurfaceCubemapLayeredArray,
		cudaDeviceProp.maxTexture1D,
		maxTexture1DLayeredArray,
		cudaDeviceProp.maxTexture1DLinear,
		cudaDeviceProp.maxTexture1DMipmap,
		maxTexture2DArray,
		maxTexture2DGatherArray,
		maxTexture2DLayeredArray,
		maxTexture2DLinearArray,

		maxTexture2DMipmapArray,
		maxTexture3DArray,
		maxTexture3DAltArray,
		cudaDeviceProp.maxTextureCubemap,
		maxTextureCubemapLayeredArray,
		maxThreadsDimArray,
		cudaDeviceProp.maxThreadsPerBlock,
		cudaDeviceProp.maxThreadsPerMultiProcessor,
		cudaDeviceProp.memPitch,
		cudaDeviceProp.memoryBusWidth,

		cudaDeviceProp.memoryClockRate,
		cudaDeviceProp.memoryPoolSupportedHandleTypes,
		cudaDeviceProp.memoryPoolsSupported,
		cudaDeviceProp.minor,
		cudaDeviceProp.multiGpuBoardGroupID,
		cudaDeviceProp.multiProcessorCount,
		env->NewStringUTF(cudaDeviceProp.name),
		cudaDeviceProp.pageableMemoryAccess,
		cudaDeviceProp.pageableMemoryAccessUsesHostPageTables,
		cudaDeviceProp.pciBusID,
		
		cudaDeviceProp.pciDeviceID,
		cudaDeviceProp.pciDomainID,
		cudaDeviceProp.persistingL2CacheMaxSize,
		cudaDeviceProp.regsPerBlock,
		
		cudaDeviceProp.regsPerMultiprocessor,
		reservedArray,
		reserved2Array,
		cudaDeviceProp.reservedSharedMemPerBlock,
		cudaDeviceProp.sharedMemPerBlock,
		cudaDeviceProp.sharedMemPerBlockOptin,
		cudaDeviceProp.sharedMemPerMultiprocessor,
		cudaDeviceProp.singleToDoublePrecisionPerfRatio,
		cudaDeviceProp.sparseCudaArraySupported,
		cudaDeviceProp.streamPrioritiesSupported,
		cudaDeviceProp.surfaceAlignment,
		cudaDeviceProp.tccDriver,
		cudaDeviceProp.textureAlignment,
		cudaDeviceProp.texturePitchAlignment,
		cudaDeviceProp.timelineSemaphoreInteropSupported,
		cudaDeviceProp.totalConstMem,
		cudaDeviceProp.totalGlobalMem,
		cudaDeviceProp.unifiedAddressing,
		cudaDeviceProp.unifiedFunctionPointers,
		uuidArray,
		cudaDeviceProp.warpSize
		);

	env->DeleteLocalRef(maxGridSizeArray);
	env->DeleteLocalRef(maxSurface1DLayeredArray);
	env->DeleteLocalRef(maxSurface2DArray);
	env->DeleteLocalRef(maxSurface2DLayeredArray);
	env->DeleteLocalRef(maxSurface3DArray);
	env->DeleteLocalRef(maxSurfaceCubemapLayeredArray);
	env->DeleteLocalRef(maxTexture1DLayeredArray);
	env->DeleteLocalRef(maxTexture2DArray);
	env->DeleteLocalRef(maxTexture2DGatherArray);
	env->DeleteLocalRef(maxTexture2DLayeredArray);
	env->DeleteLocalRef(maxTexture2DLinearArray);
	env->DeleteLocalRef(maxTexture2DMipmapArray);
	env->DeleteLocalRef(maxTexture3DArray);
	env->DeleteLocalRef(maxTexture3DAltArray);
	env->DeleteLocalRef(maxTextureCubemapLayeredArray);
	env->DeleteLocalRef(maxThreadsDimArray);
	env->DeleteLocalRef(reservedArray);
	env->DeleteLocalRef(reserved2Array);
	env->DeleteLocalRef(uuidArray);

	env->DeleteLocalRef(cudaDevicePropertiesClass);

	return cudaDevicePropertiesObject;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_initDevice(JNIEnv* env, jobject obj, jint device, jint deviceFlags, jint flags) {

	cudaError_t cudaStatus = cudaInitDevice((int)device, (unsigned int)deviceFlags, (unsigned int)flags);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_lpcCloseMemHandle(JNIEnv* env, jobject instance, jlong devicePtr) {

	cudaError_t cudaStatus = cudaIpcCloseMemHandle((void*)devicePtr);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setDevice(JNIEnv* env, jobject instance, jint device) {

	cudaError_t cudaStatus = cudaSetDevice((int)device);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setDeviceFlags(JNIEnv* env, jobject instance, jint flags) {

	cudaError_t cudaStatus = cudaSetDeviceFlags((unsigned int)flags);

	return cudaStatus;
}

JNIEXPORT jstring JNICALL Java_kuda_runtimeapi_RuntimeAPI_getErrorName(JNIEnv* env, jobject obj, jint error) {

	return env->NewStringUTF(cudaGetErrorName(static_cast<cudaError_t>(error)));
}

JNIEXPORT jstring JNICALL Java_kuda_runtimeapi_RuntimeAPI_getErrorString(JNIEnv* env, jobject obj, jint error) {

	return env->NewStringUTF(cudaGetErrorString(static_cast<cudaError_t>(error)));
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_getLastError(JNIEnv* env, jobject obj) {

	cudaError_t cudaStatus = cudaGetLastError();

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_peekAtLastError(JNIEnv* env, jobject obj) {

	cudaError_t cudaStatus = cudaPeekAtLastError();

	return cudaStatus;
}

//6.4 Stream Management
JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_ctxResetPersistingL2Cache(JNIEnv* env, jclass cls) {

	cudaError_t cudaStatus = cudaCtxResetPersistingL2Cache();

	return cudaStatus;
}

//cudaStreamAddCallback

//cudaStreamAttachMemAsync

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_beginCapture(JNIEnv* env, jclass cls, jlong stream, jint mode) {

	CUstream_st* cudaStreamPointer = reinterpret_cast<CUstream_st*>(stream);

	cudaError_t cudaStatus = cudaStreamBeginCapture(cudaStreamPointer, static_cast<cudaStreamCaptureMode>(mode));

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_copyAttributes(JNIEnv* env, jclass cls, jlong dst, jlong src) {

	CUstream_st* cudaDstStreamPointer = reinterpret_cast<CUstream_st*>(dst);

	CUstream_st* cudaSrcStreamPointer = reinterpret_cast<CUstream_st*>(src);

	cudaError_t cudaStatus = cudaStreamCopyAttributes(cudaDstStreamPointer, cudaSrcStreamPointer);

	return cudaStatus;
}

JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_StreamHandler_create(JNIEnv* env, jclass cls) {

	cudaStream_t pStream;

	cudaError_t cudaStatus = cudaStreamCreate(&pStream);


	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return (jlong)pStream;
}

JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_StreamHandler_createWithFlags(JNIEnv* env, jclass cls, jint flags) {

	cudaStream_t pStream;

	cudaError_t cudaStatus = cudaStreamCreateWithFlags(&pStream, (unsigned int)flags);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return (jlong)pStream;
}

JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_StreamHandler_createWithPriority(JNIEnv* env, jclass cls, jint flags, jint priority) {

	cudaStream_t pStream;

	cudaError_t cudaStatus = cudaStreamCreateWithPriority(&pStream, (unsigned int)flags, (int)priority);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return (jlong)pStream;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_destory(JNIEnv* env, jclass cls, jlong stream) {

	CUstream_st* cudaStreamPointer = reinterpret_cast<CUstream_st*>(stream);

	cudaError_t cudaStatus = cudaStreamDestroy(cudaStreamPointer);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_query(JNIEnv* env, jclass cls, jlong stream) {

	CUstream_st* cudaStreamPointer = reinterpret_cast<CUstream_st*>(stream);

	cudaError_t cudaStatus = cudaStreamQuery(cudaStreamPointer);

	return cudaStatus;
}

//cudaStreamSetAttribute

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_synchrnoize(JNIEnv* env, jclass cls, jlong stream) {

	CUstream_st* cudaStreamPointer = reinterpret_cast<CUstream_st*>(stream);

	cudaError_t cudaStatus = cudaStreamSynchronize(cudaStreamPointer);

	return cudaStatus;
}

//cudaStreamUpdateCaptureDependencies

//cudaStreamUpdateCaptureDependencies_v2

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_StreamHandler_waitEvent(JNIEnv* env, jclass cls, jlong stream, jlong event, jint flags) {

	CUstream_st* cudaStreamPointer = reinterpret_cast<CUstream_st*>(stream);

	CUevent_st* cudaEventPointer = reinterpret_cast<CUevent_st*>(event);

	cudaError_t cudaStatus = cudaStreamWaitEvent(cudaStreamPointer, cudaEventPointer, (unsigned int)flags);

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

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_record(JNIEnv* env, jclass cls, jlong event, jlong stream) {
	
	cudaEvent_t cudaEvent = reinterpret_cast<cudaEvent_t>(event);

	cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);

	cudaError_t cudaStatus = cudaEventRecord(cudaEvent, cudaStream);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_recordWithFlags(JNIEnv* env, jclass cls, jlong event, jlong stream, jint flags) {

	cudaEvent_t cudaEvent = reinterpret_cast<cudaEvent_t>(event);

	cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);

	cudaError_t cudaStatus = cudaEventRecordWithFlags(cudaEvent, cudaStream, (unsigned int)flags);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_synchronize(JNIEnv* env, jclass cls, jlong event) {

	CUevent_st* cudaEventPointer = reinterpret_cast<CUevent_st*>(event);

	cudaError_t cudaStatus = cudaEventSynchronize(cudaEventPointer);

	return cudaStatus;
}

//6.6 External Reource Interoperability
JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_destroyExternalMemory(JNIEnv* env, jobject obj, jlong extMem) {

	cudaExternalMemory_t cudaExternalMemoryPointer = reinterpret_cast<cudaExternalMemory_t>(extMem);

	cudaError_t cudaStatus = cudaDestroyExternalMemory(cudaExternalMemoryPointer);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_destroyExternalSemaphore(JNIEnv* env, jobject obj, jlong extSem) {

	cudaExternalSemaphore_t cudaExternalSemaphorePointer = reinterpret_cast<cudaExternalSemaphore_t>(extSem);

	cudaError_t cudaStatus = cudaDestroyExternalSemaphore(cudaExternalSemaphorePointer);

	return cudaStatus;
}

//6.9 Memory Manangement
JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_free(JNIEnv* env, jobject obj, jlong devPtr) {

	void* cudaDevPtr = reinterpret_cast<void*>(devPtr);

	cudaError_t cudaStatus = cudaFree(cudaDevPtr);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_freeArray(JNIEnv* env, jobject obj, jlong array) {

	cudaArray_t cudaArray = reinterpret_cast<cudaArray_t>(array);

	cudaError_t cudaStatus = cudaFreeArray(cudaArray);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_freeHost(JNIEnv* env, jobject obj, jlong ptr) {

	void* cudaPtr = reinterpret_cast<void*>(ptr);

	cudaError_t cudaStatus = cudaFreeHost(cudaPtr);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_freeMipmappedArray(JNIEnv* env, jobject obj, jlong mipMappedArray) {

	cudaMipmappedArray_t cudaMipMappedArray = reinterpret_cast<cudaMipmappedArray_t>(mipMappedArray);

	cudaError_t cudaStatus = cudaFreeMipmappedArray(cudaMipMappedArray);

	return cudaStatus;

}
JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_RuntimeAPI_hostAlloc(JNIEnv* env, jobject obj, jsize size, jint flags) {

	void* cudaPHost;

	cudaError_t cudaStatus = cudaHostAlloc(&cudaPHost, (size_t)size, (unsigned int)flags);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return (jlong)cudaPHost;

}

//__host__​cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int  level)
//__host__​cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol)
//__host__​cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol)


JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_RuntimeAPI_hostRegister(JNIEnv* env, jobject obj, jsize size, jint flags) {
	
	void* cudaPtr;

	cudaError_t cudaStatus = cudaHostRegister(&cudaPtr, (size_t)size, (unsigned int)flags);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return (jlong)cudaPtr;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_hostUnregister(JNIEnv* env, jobject obj, jlong ptr) {

	void* cudaPtr = reinterpret_cast<void*>(ptr);

	cudaError_t cudaStatus = cudaHostUnregister(cudaPtr);

	return cudaStatus;
}

JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_RuntimeAPI_malloc(JNIEnv* env, jobject obj, jsize size) {

	void* cudaDevPtr;

	cudaError_t cudaStatus = cudaMalloc(&cudaDevPtr, (size_t)size);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return (jlong)cudaDevPtr;
}

//__host__​cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent)
	//__host__​cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned int  flags = 0)
	//__host__​cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned int  flags = 0)

JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_RuntimeAPI_mallocHost(JNIEnv* env, jobject obj, jsize size) {
	void* cudaPtr;

	cudaError_t cudaStatus = cudaMallocHost(&cudaPtr, (size_t)size);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return (jlong)cudaPtr;
}

//6.13  Peer Device Memory Access
JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_deviceCanAccessPeer(JNIEnv* env, jobject obj, jint  device, jint  peerDevice) {
	int canAccessPeer;

	cudaError_t cudaStatus = cudaDeviceCanAccessPeer(&canAccessPeer, (int)device, (int)peerDevice);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return canAccessPeer;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_deviceDisablePeerAccess(JNIEnv* env, jobject obj, jint peerDevice) {
	
	cudaError_t cudaStatus = cudaDeviceDisablePeerAccess((int)peerDevice);

	return cudaStatus;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_deviceEnablePeerAccess(JNIEnv* env, jobject obj, jint  peerDevice, jint flags) {
	
	cudaError_t cudaStatus = cudaDeviceEnablePeerAccess((int)peerDevice, (unsigned int)flags);

	return cudaStatus;
}


//6.27 Version Management
JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_driverGetVersion(JNIEnv* env, jobject obj) {

	int driverVersion;

	cudaError_t cudaStatus = cudaDriverGetVersion(&driverVersion);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return driverVersion;
}

JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_runtimeGetVersion(JNIEnv* env, jobject instance) {

	int runtimeVersion;

	cudaError_t cudaStatus = cudaRuntimeGetVersion(&runtimeVersion);

	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	return runtimeVersion;
}
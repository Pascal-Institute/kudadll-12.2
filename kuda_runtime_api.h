#include <jni.h>

//https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

#ifdef __cplusplus
extern "C" {
#endif
	//1. Device Management
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_chooseDevice(JNIEnv* env, jclass cls, jobject cudaDeviceProp);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_flushGPUDirectRDMAWrites(JNIEnv* env, jclass cls, jint scope);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getAttribute(JNIEnv* env, jclass cls, jint cudaDeviceAttr, jint device);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_DeviceHandler_getDefaultMemPool(JNIEnv* env, jclass cls, jint  device);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getLimit(JNIEnv* env, jclass cls, jbyte limit);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_DeviceHandler_getMemPool(JNIEnv* env, jclass cls, jint  device);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getP2PAttribute(JNIEnv* env, jclass cls, jint attr, jint scrDevice, jint dstDevice);

	JNIEXPORT jstring JNICALL Java_kuda_runtimeapi_DeviceHandler_getPCIBusId(JNIEnv* env, jclass cls, jint device);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getStreamPriorityRange(JNIEnv* env, jclass cls);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setCacheConfig(JNIEnv* env, jclass cls, jint cacheConfig);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setLimit(JNIEnv* env, jclass cls, jbyte limit, jsize value);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setSharedMemConfig(JNIEnv* env, jclass cls, jint config);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_synchronize(JNIEnv* env, jclass cls);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_reset(JNIEnv* env, jclass cls);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getDevice(JNIEnv* env, jclass cls);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getDiviceCount(JNIEnv* env, jclass cls);

	JNIEXPORT jobject JNICALL Java_kuda_runtimeapi_DeviceHandler_getDeviceProperties(JNIEnv* env, jclass cls, jint device);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_initDevice(JNIEnv* env, jclass cls, jint device, jint deviceFlags, jint flags);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_lpcCloseMemHandle(JNIEnv* env, jclass cls, jlong devicePtr);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_lpcCloseMemHandle(JNIEnv* env, jclass cls, jlong devicePtr);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setDevice(JNIEnv* env, jclass cls, jint device);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setDeviceFlags(JNIEnv* env, jclass cls, jint flags);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setValidDevices(JNIEnv* env, jclass cls, jint len);

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

	//27. Version Management
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_driverGetVersion(JNIEnv* env, jobject obj);
	
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_RuntimeAPI_runtimeGetVersion(JNIEnv* env, jobject obj);

	//28. Graph Management
	//__host__​cudaError_t cudaDeviceGetGraphMemAttribute(int  device, cudaGraphMemAttributeType attr, void* value)
	//__host__​cudaError_t cudaDeviceGraphMemTrim(int  device)
	//__host__​cudaError_t cudaDeviceSetGraphMemAttribute(int  device, cudaGraphMemAttributeType attr, void* value)
	//__device__​cudaGraphExec_t 	cudaGetCurrentGraphExec(void)
	//__host__​cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph)
	//__host__​cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies)
	//__host__​cudaError_t cudaGraphAddDependencies_v2(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, const cudaGraphEdgeData * edgeData, size_t numDependencies)
	//__host__​cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies)
	//__host__​cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event)
	//__host__​cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event)
	//__host__​cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams)
	//__host__​cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams)
	//__host__​cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaMemAllocNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void* dptr)
	//__host__​cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams)
	//__host__​cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind)
	//__host__​cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind)
	//__host__​cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind)
	//__host__​cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams)
	//__host__​cudaError_t cudaGraphAddNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraphNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphAddNode_v2(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, const cudaGraphEdgeData * dependencyData, size_t numDependencies, cudaGraphNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph)
	//__host__​cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph)
	//__host__​cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle * pHandle_out, cudaGraph_t graph, unsigned int  defaultLaunchValue = 0, unsigned int  flags = 0)
	//__host__​cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned int  flags)
	//__host__​cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int  flags)
	//__host__​cudaError_t cudaGraphDestroy(cudaGraph_t graph)
	//__host__​cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node)
	//__host__​cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out)
	//__host__​cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event)
	//__host__​cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out)
	//__host__​cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event)
	//__host__​cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph)
	//__host__​cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec)
	//__host__​cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event)
	//__host__​cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event)
	//__host__​cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags)
	//__host__​cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams)
	//__host__​cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams)
	//__host__​cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams)
	//__host__​cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind)
	//__host__​cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind)
	//__host__​cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind)
	//__host__​cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams * pNodeParams)
	//__host__​cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo * resultInfo)
	//__host__​cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams * params_out)
	//__host__​cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams * params_out)
	//__host__​cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges)
	//__host__​cudaError_t cudaGraphGetEdges_v2(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, cudaGraphEdgeData * edgeData, size_t * numEdges)
	//__host__​cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes)
	//__host__​cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes)
	//__host__​cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams)
	//__host__​cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams)
	//__host__​cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned long long flags = 0)
	//__host__​cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned long long flags = 0)
	//__host__​cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams * instantiateParams)
	//__host__​cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst)
	//__host__​cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue * value_out)
	//__host__​cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams)
	//__host__​cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue * value)
	//__host__​cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams)
	//__host__​__device__​cudaError_t 	cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
	//__host__​cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams * params_out)
	//__host__​cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out)
	//__host__​cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams)
	//__host__​cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams)
	//__host__​cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind)
	//__host__​cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind)
	//__host__​cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind)
	//__host__​cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams)
	//__host__​cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams)
	//__host__​cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph)
	//__host__​cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies)
	//__host__​cudaError_t cudaGraphNodeGetDependencies_v2(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, cudaGraphEdgeData * edgeData, size_t * pNumDependencies)
	//__host__​cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes)
	//__host__​cudaError_t cudaGraphNodeGetDependentNodes_v2(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, cudaGraphEdgeData * edgeData, size_t * pNumDependentNodes)
	//__host__​cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled)
	//__host__​cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * *pType)
	//__host__​cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int  isEnabled)
	//__host__​cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams * nodeParams)
	//__host__​cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int  count = 1)
	//__host__​cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies)
	//__host__​cudaError_t cudaGraphRemoveDependencies_v2(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, const cudaGraphEdgeData * edgeData, size_t numDependencies)
	//__host__​cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int  count = 1, unsigned int  flags = 0)
	//__device__​ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int  value)
	//__host__​cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream)
	//__host__​cudaError_t cudaUserObjectCreate(cudaUserObject_t * object_out, void* ptr, cudaHostFn_t destroy, unsigned int  initialRefcount, unsigned int  flags)
	//__host__​cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int  count = 1)
	//__host__​cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int  count = 1)
#ifdef __cplusplus
}
#endif
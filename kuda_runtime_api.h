#include <jni.h>

//https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

#ifdef __cplusplus
extern "C" {
#endif
	//6.1 Device Management
	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getLimit(JNIEnv* env, jclass cls, jbyte limit);

	JNIEXPORT jstring JNICALL Java_kuda_runtimeapi_DeviceHandler_getPCIBusId(JNIEnv* env, jclass cls, jint device);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_getStreamPriorityRange(JNIEnv* env, jclass cls);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setCacheConfig(JNIEnv* env, jclass cls, jint cacheConfig);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setLimit(JNIEnv* env, jclass cls, jbyte limit, jsize value);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_setSharedMemConfig(JNIEnv* env, jclass cls, jint config);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_synchronize(JNIEnv* env, jclass cls);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_DeviceHandler_reset(JNIEnv* env, jclass cls);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_runtimeGetVersion(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_getDevice(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_getDiviceCount(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_initDevice(JNIEnv* env, jobject obj, jint device, jint deviceFlags, jint flags);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_lpcCloseMemHandle(JNIEnv* env, jobject obj, jlong devicePtr);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_setDevice(JNIEnv* env, jobject obj, jint device);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_setDeviceFlags(JNIEnv* env, jobject obj, jint flags);

	//6.3 Error Handling
	JNIEXPORT jstring JNICALL Java_kuda_RuntimeAPI_getErrorName(JNIEnv* env, jobject obj, jint error);

	JNIEXPORT jstring JNICALL Java_kuda_RuntimeAPI_getErrorString(JNIEnv* env, jobject obj, jint error);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_getLastError(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_peekAtLastError(JNIEnv* env, jobject obj);

	//6.4 Stream Management
	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_ctxResetPersistingL2Cache(JNIEnv* env, jobject obj);

	//cudaStreamAddCallback

	//cudaStreamAttachMemAsync

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_streamBeginCapture(JNIEnv* env, jobject obj, jlong stream, jint mode);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_streamBeginCapture(JNIEnv* env, jobject obj, jlong stream, jint mode);

	//cudaStreamBeginCaptureToGraph

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_streamCopyAttributes(JNIEnv* env, jobject obj, jlong dst, jlong src);

	JNIEXPORT jlong JNICALL Java_kuda_RuntimeAPI_streamCreate(JNIEnv* env, jobject obj);

	JNIEXPORT jlong JNICALL Java_kuda_RuntimeAPI_streamCreateWithFlags(JNIEnv* env, jobject obj, jint flags);

	JNIEXPORT jlong JNICALL Java_kuda_RuntimeAPI_streamCreateWithPriority(JNIEnv* env, jobject obj, jint flags, jint priority);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_streamDestory(JNIEnv* env, jobject obj, jlong stream);


	//6.5 Event ManageMent
	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_EventHandler_create(JNIEnv* env, jclass cls);

	JNIEXPORT jlong JNICALL Java_kuda_runtimeapi_EventHandler_createWithFlags(JNIEnv* env, jclass cls, jint flags);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_destroy(JNIEnv* env, jclass cls, jlong event);

	JNIEXPORT jfloat JNICALL Java_kuda_runtimeapi_EventHandler_elapsedTime(JNIEnv* env, jclass cls, jlong start, jlong end);

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_query(JNIEnv* env, jclass cls, jlong event);

	//cudaEventRecord

	//cudaEventRecordWithFlags

	JNIEXPORT jint JNICALL Java_kuda_runtimeapi_EventHandler_synchronize(JNIEnv* env, jclass cls, jlong event);

#ifdef __cplusplus
}
#endif
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

	JNIEXPORT jlong JNICALL Java_kuda_RuntimeAPI_streamCreate(JNIEnv* env, jobject obj);

	JNIEXPORT jlong JNICALL Java_kuda_RuntimeAPI_streamCreateWithFlags(JNIEnv* env, jobject obj, jint flags);

	//6.5 Event ManageMent
	JNIEXPORT jlong JNICALL Java_kuda_RuntimeAPI_eventCreate(JNIEnv* env, jobject obj);

	JNIEXPORT jlong JNICALL Java_kuda_RuntimeAPI_eventCreateWithFlags(JNIEnv* env, jobject obj, jint flags);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_eventDestroy(JNIEnv* env, jobject obj, jlong event);

	JNIEXPORT jfloat JNICALL Java_kuda_RuntimeAPI_eventElapsedTime(JNIEnv* env, jobject obj, jlong start, jlong end);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_eventQuery(JNIEnv* env, jobject obj, jlong event);

	//cudaEventRecord

	//cudaEventRecordWithFlags

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_eventSynchronize(JNIEnv* env, jobject obj, jlong event);

#ifdef __cplusplus
}
#endif
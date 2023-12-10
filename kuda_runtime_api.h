#include <jni.h>

//https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

#ifdef __cplusplus
extern "C" {
#endif
	//6.1 Device Management
	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceGetLimit(JNIEnv* env, jobject obj, jbyte limit);

	JNIEXPORT jstring JNICALL Java_kuda_RuntimeAPI_deviceGetPCIBusId(JNIEnv* env, jobject obj, jint device);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceGetStreamPriorityRange(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceSetCacheConfig(JNIEnv* env, jobject obj, jint cacheConfig);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceSetLimit(JNIEnv* env, jobject obj, jbyte limit, jsize value);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceSetSharedMemConfig(JNIEnv* env, jobject obj, jint config);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceSynchronize(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_deviceReset(JNIEnv* env, jobject obj);

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

	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_streamCreate(JNIEnv* env, jobject obj);

	//6.5 Event ManageMent
	JNIEXPORT jint JNICALL Java_kuda_RuntimeAPI_eventCreate(JNIEnv* env, jobject obj, jobject event);

#ifdef __cplusplus
}
#endif
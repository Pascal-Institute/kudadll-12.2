#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_syncDevice(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_resetDevice(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_getRuntimeVersion(JNIEnv* env, jobject obj);
	
	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_getDivice(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_getDiviceCount(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_setDevice(JNIEnv* env, jobject obj, jint device);

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_setDeviceFlags(JNIEnv* env, jobject obj, jint flags);
#ifdef __cplusplus
}
#endif
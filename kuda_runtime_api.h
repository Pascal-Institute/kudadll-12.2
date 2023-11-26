#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_syncDevice(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_getRuntimeVersion(JNIEnv* env, jobject obj);
	
	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_getDivice(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_getDiviceCount(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_setDevice(JNIEnv* env, jobject obj, jint device);

#ifdef __cplusplus
}
#endif
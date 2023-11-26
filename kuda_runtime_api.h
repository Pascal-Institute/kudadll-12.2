#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_getRuntimeVersion(JNIEnv* env, jobject obj);
	
	JNIEXPORT jint JNICALL Java_KudaRuntimeAPI_getDivice(JNIEnv* env, jobject obj);

#ifdef __cplusplus
}
#endif
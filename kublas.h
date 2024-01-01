#include <jni.h>

// https://docs.nvidia.com/cuda/cublas/index.html

#ifdef __cplusplus
extern "C" {
#endif
	JNIEXPORT jlong JNICALL Java_kuda_Kublas_create(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_Kublas_destroy(JNIEnv* env, jobject obj, jlong handle);

	JNIEXPORT jint JNICALL Java_kuda_Kublas_getVersion(JNIEnv* env, jobject obj, jlong handle);

	JNIEXPORT jint JNICALL Java_kuda_Kublas_getProperty(JNIEnv* env, jobject obj, jint type);

	JNIEXPORT jstring JNICALL Java_kuda_Kublas_getStatusName(JNIEnv* env, jobject obj, jint status);

	JNIEXPORT jstring JNICALL Java_kuda_Kublas_getStatusString(JNIEnv* env, jobject obj, jint status);

	JNIEXPORT jint JNICALL Java_kuda_Kublas_setStream(JNIEnv* env, jobject obj, jlong handle, jlong streamId);
#ifdef __cplusplus
}
#endif
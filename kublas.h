#include <jni.h>

// https://docs.nvidia.com/cuda/cublas/index.html

#ifdef __cplusplus
extern "C" {
#endif
	JNIEXPORT jlong JNICALL Java_kuda_Kublas_create(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_Kublas_destroy(JNIEnv* env, jobject obj, jlong handle);

	JNIEXPORT jint JNICALL Java_kuda_Kublas_getVersion(JNIEnv* env, jobject obj, jlong handle);

	JNIEXPORT jint JNICALL Java_kuda_Kublas_getProperty(JNIEnv* env, jobject obj, jint type);
#ifdef __cplusplus
}
#endif
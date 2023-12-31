#include <jni.h>

// https://docs.nvidia.com/cuda/cublas/index.html

#ifdef __cplusplus
extern "C" {
#endif
	JNIEXPORT jlong JNICALL Java_kuda_KublasAPI_create(JNIEnv* env, jobject obj);

#ifdef __cplusplus
}
#endif
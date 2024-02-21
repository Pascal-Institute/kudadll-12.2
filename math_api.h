#include <jni.h>

//https://docs.nvidia.com/cuda/archive/12.2.2/cuda-math-api/index.html

#ifdef __cplusplus
extern "C" {
#endif
	JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_log(JNIEnv* env, jobject obj, jdouble x);
#ifdef __cplusplus
}
#endif
#include <jni.h>

//https://docs.nvidia.com/cuda/cuda-driver-api/index.html

#ifdef __cplusplus
extern "C" {
#endif
	//6.3 Initialization
	JNIEXPORT jint JNICALL Java_kuda_DriverAPI_init(JNIEnv* env, jobject obj, jint flags);

#ifdef __cplusplus
}
#endif
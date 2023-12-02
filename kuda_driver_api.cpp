#include "kuda_driver_api.h"
#include <jni.h>
#include <cuda.h>

JNIEXPORT jint JNICALL Java_kuda_DriverAPI_init(JNIEnv* env, jobject obj, jint flags) {
	
	CUresult cudaStatus = cuInit((unsigned int) flags);

	return cudaStatus;
}

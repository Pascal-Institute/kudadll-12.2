#include "kublas.h"
#include <jni.h>
#include <cublas.h>

JNIEXPORT jlong JNICALL Java_kuda_Kublas_create(JNIEnv* env, jobject obj) {
	cublasHandle_t handle;

	cublasStatus_t cublasStatus = cublasCreate_v2(&handle);
	
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		return cublasStatus;
	}
	
	return (jlong) handle;
}
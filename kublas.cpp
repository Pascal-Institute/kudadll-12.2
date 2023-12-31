#include "kublas.h"
#include <jni.h>
#include <cublas_v2.h>

JNIEXPORT jlong JNICALL Java_kuda_Kublas_create(JNIEnv* env, jobject obj) {
	cublasHandle_t handle;

	cublasStatus_t cublasStatus = cublasCreate(&handle);
	
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		return cublasStatus;
	}
	
	return (jlong) handle;
}

JNIEXPORT jint JNICALL Java_kuda_Kublas_destroy(JNIEnv* env, jobject obj, jlong handle) {
	
	cublasHandle_t cublasHandle = reinterpret_cast<cublasHandle_t>(handle);

	cublasStatus_t cublasStatus = cublasDestroy(cublasHandle);

	return cublasStatus;
}
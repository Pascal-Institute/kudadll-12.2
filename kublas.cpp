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


JNIEXPORT jint JNICALL Java_kuda_Kublas_getVersion(JNIEnv* env, jobject obj, jlong handle) {

	int version;

	cublasHandle_t cublasHandle = reinterpret_cast<cublasHandle_t>(handle);

	cublasStatus_t cublasStatus = cublasGetVersion(cublasHandle, &version);

	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		return cublasStatus;
	}

	return version;
}

JNIEXPORT jint JNICALL Java_kuda_Kublas_getProperty(JNIEnv* env, jobject obj, jint type) {
	
	int version;

	libraryPropertyType_t libraryPropertyType = static_cast<libraryPropertyType_t>(type);

	cublasStatus_t cublasStatus = cublasGetProperty(libraryPropertyType, &version);

	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		return cublasStatus;
	}

	return version;
}

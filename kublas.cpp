#include "kublas.h"
#include <jni.h>
#include <cublas_v2.h>

JNIEXPORT jlong JNICALL Java_kuda_kublas_Kublas_create(JNIEnv* env, jobject obj) {
	cublasHandle_t handle;

	cublasStatus_t cublasStatus = cublasCreate(&handle);
	
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		return cublasStatus;
	}
	
	return (jlong) handle;
}

JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_destroy(JNIEnv* env, jobject obj, jlong handle) {
	
	cublasHandle_t cublasHandle = reinterpret_cast<cublasHandle_t>(handle);

	cublasStatus_t cublasStatus = cublasDestroy(cublasHandle);

	return cublasStatus;
}


JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_getVersion(JNIEnv* env, jobject obj, jlong handle) {

	int version;

	cublasHandle_t cublasHandle = reinterpret_cast<cublasHandle_t>(handle);

	cublasStatus_t cublasStatus = cublasGetVersion(cublasHandle, &version);

	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		return cublasStatus;
	}

	return version;
}

JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_getProperty(JNIEnv* env, jobject obj, jint type) {
	
	int version;

	libraryPropertyType_t libraryPropertyType = static_cast<libraryPropertyType_t>(type);

	cublasStatus_t cublasStatus = cublasGetProperty(libraryPropertyType, &version);

	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		return cublasStatus;
	}

	return version;
}

JNIEXPORT jstring JNICALL Java_kuda_kublas_Kublas_getStatusName(JNIEnv* env, jobject obj, jint status) {

	return env->NewStringUTF(cublasGetStatusName(static_cast<cublasStatus_t>(status)));
}

JNIEXPORT jstring JNICALL Java_kuda_kublas_Kublas_getStatusString(JNIEnv* env, jobject obj, jint status) {

	return env->NewStringUTF(cublasGetStatusString(static_cast<cublasStatus_t>(status)));
}

JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_setStream(JNIEnv* env, jobject obj, jlong handle, jlong streamId) {

	cublasHandle_t cublasHandle = reinterpret_cast<cublasHandle_t>(handle);

	cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(streamId);

	cublasStatus_t cublasStatus = cublasSetStream(cublasHandle, cudaStream);

	return cublasStatus;
}

JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_setWorkspace(JNIEnv* env, jobject obj, jlong handle, jsize workspaceSizeInBytes) {

	void* workspace{};

	cublasHandle_t cublasHandle = reinterpret_cast<cublasHandle_t>(handle);

	cublasStatus_t cublasStatus = cublasSetWorkspace(cublasHandle, workspace, (size_t)workspaceSizeInBytes);

	return cublasStatus;
}

JNIEXPORT jlong JNICALL Java_kuda_kublas_Kublas_getStream(JNIEnv* env, jobject obj, jlong handle) {

	cudaStream_t cudaStream;

	cublasHandle_t cublasHandle = reinterpret_cast<cublasHandle_t>(handle);

	cublasStatus_t cublasStatus = cublasGetStream(cublasHandle, &cudaStream);

	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		return cublasStatus;
	}

	return (jlong)cudaStream;
}

JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_getPointerMode(JNIEnv* env, jobject obj, jlong handle) {

	//initialize...
	cublasPointerMode_t cublasPointerMode = CUBLAS_POINTER_MODE_HOST;

	cublasHandle_t cublasHandle = reinterpret_cast<cublasHandle_t>(handle);

	cublasStatus_t cublasStatus = cublasGetPointerMode(cublasHandle, &cublasPointerMode);

	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		return cublasStatus;
	}

	return (jint)static_cast<int>(cublasPointerMode);
}

JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_setPointerMode(JNIEnv* env, jobject obj, jlong handle, jint mode) {

	cublasHandle_t cublasHandle = reinterpret_cast<cublasHandle_t>(handle);

	cublasStatus_t cublasStatus = cublasSetPointerMode(cublasHandle, static_cast<cublasPointerMode_t>(mode));

	return cublasStatus;
}

JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_setVector(JNIEnv* env, jobject obj, jint n, jint elemSize, jlong x, jint incx, jlong y, jint incy) {
	
	const void* cublasX = reinterpret_cast<void*>(x);

	void* cublasY = reinterpret_cast<void*>(y);

	cublasStatus_t cublasStatus = cublasSetVector(n, elemSize, cublasX, incx, cublasY, incy);

	return cublasStatus;
}

JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_getVector(JNIEnv* env, jobject obj, jint n, jint elemSize, jlong x, jint incx, jlong y, jint incy) {

	const void* cublasX = reinterpret_cast<void*>(x);

	void* cublasY = reinterpret_cast<void*>(y);

	cublasStatus_t cublasStatus = cublasGetVector(n, elemSize, cublasX, incx, cublasY, incy);

	return cublasStatus;
}

JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_setMatrix(JNIEnv* env, jobject obj, jint rows, jint cols, jint elemSize, jlong A, jint lda, jlong B, jint ldb) {
	
	const void* cublasA = reinterpret_cast<void*>(A);

	void* cublasB = reinterpret_cast<void*>(B);

	cublasStatus_t cublasStatus = cublasSetMatrix(rows, cols, elemSize, cublasA, lda, cublasB, ldb);

	return cublasStatus;
}

JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_getMatrix(JNIEnv* env, jobject obj, jint rows, jint cols, jint elemSize, jlong A, jint lda, jlong B, jint ldb) {

	const void* cublasA = reinterpret_cast<void*>(A);

	void* cublasB = reinterpret_cast<void*>(B);

	cublasStatus_t cublasStatus = cublasGetMatrix(rows, cols, elemSize, cublasA, lda, cublasB, ldb);

	return cublasStatus;
}
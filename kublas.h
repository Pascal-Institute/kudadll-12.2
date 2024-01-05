#include <jni.h>

// https://docs.nvidia.com/cuda/cublas/index.html

#ifdef __cplusplus
extern "C" {
#endif
	JNIEXPORT jlong JNICALL Java_kuda_kublas_Kublas_create(JNIEnv* env, jobject obj);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_destroy(JNIEnv* env, jobject obj, jlong handle);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_getVersion(JNIEnv* env, jobject obj, jlong handle);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_getProperty(JNIEnv* env, jobject obj, jint type);

	JNIEXPORT jstring JNICALL Java_kuda_kublas_Kublas_getStatusName(JNIEnv* env, jobject obj, jint status);

	JNIEXPORT jstring JNICALL Java_kuda_kublas_Kublas_getStatusString(JNIEnv* env, jobject obj, jint status);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_setStream(JNIEnv* env, jobject obj, jlong handle, jlong streamId);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_setWorkspace(JNIEnv* env, jobject obj, jlong handle, jsize workspaceSizeInBytes);

	JNIEXPORT jlong JNICALL Java_kuda_kublas_Kublas_getStream(JNIEnv* env, jobject obj, jlong handle);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_getPointerMode(JNIEnv* env, jobject obj, jlong handle);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_setPointerMode(JNIEnv* env, jobject obj, jlong handle, jint mode);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_setVector(JNIEnv* env, jobject obj, jint n, jint elemSize, jlong x , jint incx, jlong y, jint incy);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_getVector(JNIEnv* env, jobject obj, jint n, jint elemSize, jlong x, jint incx, jlong y, jint incy);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_setMatrix(JNIEnv* env, jobject obj, jint rows, jint cols, jint elemSize, jlong A, jint lda, jlong B, jint ldb);

	JNIEXPORT jint JNICALL Java_kuda_kublas_Kublas_getMatrix(JNIEnv* env, jobject obj, jint rows, jint cols, jint elemSize, jlong A, jint lda, jlong B, jint ldb);
#ifdef __cplusplus
}
#endif
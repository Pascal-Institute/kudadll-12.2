#include <jni.h>

//https://docs.nvidia.com/cuda/cuda-driver-api/index.html

#ifdef __cplusplus
extern "C" {
#endif
	//6.3 Initialization
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_init(JNIEnv* env, jobject obj, jint flags);

	//6.4 Version Management
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_getDriverVersion(JNIEnv* env, jobject obj);

	//6.5 Device Management
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_getDevice(JNIEnv* env, jobject obj, jint ordinal);

	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_getDeviceCount(JNIEnv* env, jobject obj);

#ifdef __cplusplus
}
#endif
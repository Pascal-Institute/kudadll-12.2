#include <jni.h>

//https://docs.nvidia.com/cuda/cuda-driver-api/index.html

#ifdef __cplusplus
extern "C" {
#endif

	//2. Error Handling
	JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_getErrorName(JNIEnv* env, jobject obj, jint error);

	JNIEXPORT jstring JNICALL Java_kuda_driverapi_DriverAPI_getErrorString(JNIEnv* env, jobject obj, jint error);

	//3. Initialization
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_init(JNIEnv* env, jobject obj, jint flags);

	//4. Version Management
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_getDriverVersion(JNIEnv* env, jobject obj);

	//5. Device Management
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_deviceGet(JNIEnv* env, jobject obj, jint ordinal);

	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_deviceGetCount(JNIEnv* env, jobject obj);

	//7. Context Management
	//CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)
	JNIEXPORT jint JNICALL Java_kuda_driverapi_DriverAPI_devicePrimaryCtxRelease(JNIEnv* env, jobject obj, jint device);
	//CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
	//CUresult cuDevicePrimaryCtxReset(CUdevice dev)
	//CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev)
	//CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int  flags)

#ifdef __cplusplus
}
#endif
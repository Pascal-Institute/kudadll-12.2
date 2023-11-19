#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

	JNIEXPORT jint JNICALL Java_com_example_MyClass_nativeMethod(JNIEnv* env, jobject obj);

#ifdef __cplusplus
}
#endif
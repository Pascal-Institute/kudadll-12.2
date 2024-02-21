#include "math_api.h"
#include <jni.h>
#include <vcruntime_string.h>
#include <cuda/std/cmath>
#define __CUDA_INTERNAL_COMPILATION__
#include <crt/math_functions.hpp>
#undef __CUDA_INTERNAL_COMPILATION__

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_log(JNIEnv* env, jobject obj, jdouble x) {
	return log(x);
}
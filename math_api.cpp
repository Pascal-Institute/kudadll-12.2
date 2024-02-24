#include "math_api.h"
#include <jni.h>
#include <vcruntime_string.h>
#include <cuda/std/cmath>
#include <cmath>
#define __CUDA_INTERNAL_COMPILATION__
#include <crt/math_functions.hpp>
#undef __CUDA_INTERNAL_COMPILATION__

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_acos(JNIEnv* env, jclass cls, jdouble x) {
	return acos(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_acosh(JNIEnv* env, jclass cls, jdouble x){
	return acosh(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_asin(JNIEnv* env, jclass cls, jdouble x) {
	return asin(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_asinh(JNIEnv* env, jclass cls, jdouble x) {
	return asinh(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_atan(JNIEnv* env, jclass cls, jdouble x) {
	return atan(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_atan2(JNIEnv* env, jclass cls, jdouble y, jdouble x) {
	return atan2(y, x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_atanh(JNIEnv* env, jclass cls, jdouble x) {
	return atanh(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_cbrt(JNIEnv* env, jclass cls, jdouble x) {
	return cbrt(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_ceil(JNIEnv* env, jclass cls, jdouble x) {
	return ceil(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_copysign(JNIEnv* env, jclass cls, jdouble x, jdouble y) {
	return copysign(x, y);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_cos(JNIEnv* env, jclass cls, jdouble x) {
	return cos(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_cosh(JNIEnv* env, jclass cls, jdouble x) {
	return cosh(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_cospi(JNIEnv* env, jclass cls, jdouble x) {
	return cospi(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_erf(JNIEnv* env, jclass cls, jdouble x) {
    return erf(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_erfc(JNIEnv* env, jclass cls, jdouble x) {
    return erfc(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_erfcinv(JNIEnv* env, jclass cls, jdouble x) {
    return erfcinv(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_erfcx(JNIEnv* env, jclass cls, jdouble x) {
    return erfcx(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_erfinv(JNIEnv* env, jclass cls, jdouble x) {
    return erfinv(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_exp(JNIEnv* env, jclass cls, jdouble x) {
    return exp(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_exp10(JNIEnv* env, jclass cls, jdouble x) {
    return exp10(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_exp2(JNIEnv* env, jclass cls, jdouble x) {
    return exp2(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_expm1(JNIEnv* env, jclass cls, jdouble x) {
    return expm1(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_fabs(JNIEnv* env, jclass cls, jdouble x) {
    return fabs(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_fdim(JNIEnv* env, jclass cls, jdouble x, jdouble y) {
    return fdim(x, y);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_floor(JNIEnv* env, jclass cls, jdouble x) {
    return floor(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_fma(JNIEnv* env, jclass cls, jdouble x, jdouble y, jdouble z) {
    return fma(x, y, z);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_fmax(JNIEnv* env, jclass cls, jdouble x, jdouble y) {
    return fmax(x, y);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_fmin(JNIEnv* env, jclass cls, jdouble x, jdouble y) {
    return fmin(x, y);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_fmod(JNIEnv* env, jclass cls, jdouble x, jdouble y) {
    return fmod(x, y);
}
//double frexp(double  x, int* nptr)

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_hypot(JNIEnv* env, jclass cls, jdouble x, jdouble y) {
    return hypot(x, y);
}

JNIEXPORT jint JNICALL Java_kuda_mathapi_MathAPI_ilogb(JNIEnv* env, jclass cls, jdouble x) {
    return ilogb(x);
}

//__RETURN_TYPE 	isfinite(double  a)
//__RETURN_TYPE 	isinf(double  a)
//__RETURN_TYPE 	isnan(double  a)

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_j0(JNIEnv* env, jclass cls, jdouble x) {
    return j0(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_j1(JNIEnv* env, jclass cls, jdouble x) {
    return j1(x);
}

//double jn(int  n, double  x)
//double ldexp(double  x, int  exp)
//double lgamma(double  x)
//long long int 	llrint(double  x)
//long long int 	llround(double  x)

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_log(JNIEnv* env, jclass cls, jdouble x) {
	return log(x);
}

//double log10(double  x)
//double log1p(double  x)
//double log2(double  x)
//double logb(double  x)
//long int lrint(double  x)
//long int lround(double  x)
//double max(const double  a, const float  b)
//double max(const float  a, const double  b)
//double max(const double  a, const double  b)
//double min(const double  a, const float  b)
//double min(const float  a, const double  b)
//double min(const double  a, const double  b)
//double modf(double  x, double* iptr)
//double nan(const char* tagp)
//double nearbyint(double  x)
//double nextafter(double  x, double  y)
//double norm(int  dim, const double* p)
//double norm3d(double  a, double  b, double  c)
//double norm4d(double  a, double  b, double  c, double  d)
//double normcdf(double  x)
//double normcdfinv(double  x)
//double pow(double  x, double  y)
//double rcbrt(double  x)
//double remainder(double  x, double  y)
//double remquo(double  x, double  y, int* quo)
//double rhypot(double  x, double  y)
//double rint(double  x)
//double rnorm(int  dim, const double* p)
//double rnorm3d(double  a, double  b, double  c)
//double rnorm4d(double  a, double  b, double  c, double  d)
//double round(double  x)
//double rsqrt(double  x)
//double scalbln(double  x, long int  n)
//double scalbn(double  x, int  n)
//__RETURN_TYPE 	signbit(double  a)
//double sin(double  x)
//void sincos(double  x, double* sptr, double* cptr)
//void sincospi(double  x, double* sptr, double* cptr)
//double sinh(double  x)
//double sinpi(double  x)
//double sqrt(double  x)
//double tan(double  x)
//double tanh(double  x)
//double tgamma(double  x)
//double trunc(double  x)
//double y0(double  x)
//double y1(double  x)
//double yn(int  n, double  x)
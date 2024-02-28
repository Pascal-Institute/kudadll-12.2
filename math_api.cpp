#include "math_api.h"
#include <jni.h>
#include <cmath>
#include <vcruntime_string.h>
#include <cuda/std/cmath>
#define __CUDA_INTERNAL_COMPILATION__
#include <crt/math_functions.h>
#include <crt/math_functions.hpp>
#undef __CUDA_INTERNAL_COMPILATION__

//DEPRECATED FUNCTION
int max(int a, int b) {
    return (a > b) ? a : b;
}

//DEPRECATED FUNCTION
int min(int a, int b) {
    return (a < b) ? a : b;
}


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
    return _j0(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_j1(JNIEnv* env, jclass cls, jdouble x) {
    return _j1(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_jn(JNIEnv* env, jclass cls, jint n, jdouble x) {
    return _jn(n, x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_ldexp(JNIEnv* env, jclass cls, jdouble x, jint exp) {
    return ldexp(x, exp);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_lgamma(JNIEnv* env, jclass cls, jdouble x) {
    return lgamma(x);
}

JNIEXPORT jlong JNICALL Java_kuda_mathapi_MathAPI_llrint(JNIEnv* env, jclass cls, jdouble x) {
    return llrint(x);
}

JNIEXPORT jlong JNICALL Java_kuda_mathapi_MathAPI_llround(JNIEnv* env, jclass cls, jdouble x) {
    return llround(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_log(JNIEnv* env, jclass cls, jdouble x) {
	return log(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_log10(JNIEnv* env, jclass cls, jdouble x) {
    return log10(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_log1p(JNIEnv* env, jclass cls, jdouble x) {
    return log1p(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_log2(JNIEnv* env, jclass cls, jdouble x) {
    return log2(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_logb(JNIEnv* env, jclass cls, jdouble x) {
    return logb(x);
}

JNIEXPORT jlong JNICALL Java_kuda_mathapi_MathAPI_lrint(JNIEnv* env, jclass cls, jdouble x) {
    return lrint(x);
}

JNIEXPORT jlong JNICALL Java_kuda_mathapi_MathAPI_lround(JNIEnv* env, jclass cls, jdouble x) {
    return lround(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_max1(JNIEnv* env, jclass cls, jdouble a, jfloat b) {
  
    return max((const double)a, (const float)b);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_max2(JNIEnv* env, jclass cls, jfloat a, jdouble b) {

    return max((const float)a, (const double)b);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_max(JNIEnv* env, jclass cls, jdouble a, jdouble b) {

    return max((const double)a, (const double)b);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_min1(JNIEnv* env, jclass cls, jdouble a, jfloat b) {
    return min((const double)a, (const float)b);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_min2(JNIEnv* env, jclass cls, jfloat a, jdouble b) {
    return min((const float)a, (const double)b);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_min(JNIEnv* env, jclass cls, jdouble a, jdouble b) {
    return min((const double)a, (const double)b);
}

//double modf(double  x, double* iptr)
//double nan(const char* tagp)

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_nearbyint(JNIEnv* env, jclass cls, jdouble x) {
    return nearbyint(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_nextafter(JNIEnv* env, jclass cls, jdouble x, jdouble y) {
    return nextafter(x, y);
}

//double norm(int  dim, const double* p)
//double norm3d(double  a, double  b, double  c)
//double norm4d(double  a, double  b, double  c, double  d)
//double normcdf(double  x)
//double normcdfinv(double  x)

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_pow(JNIEnv* env, jclass cls, jdouble x, jdouble y) {
    return pow(x, y);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_rcbrt(JNIEnv* env, jclass cls, jdouble x) {
    return rcbrt(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_remainder(JNIEnv* env, jclass cls, jdouble x, jdouble y) {
    return remainder(x, y);
}

//double remquo(double  x, double  y, int* quo)
//double rhypot(double  x, double  y)
//double rint(double  x)
//double rnorm(int  dim, const double* p)
//double rnorm3d(double  a, double  b, double  c)
//double rnorm4d(double  a, double  b, double  c, double  d)

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_round(JNIEnv* env, jclass cls, jdouble x) {
    return round(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_rsqrt(JNIEnv* env, jclass cls, jdouble x) {
    return rsqrt(x);
}

JNIEXPORT jdouble JNICALL Java_kuda_mathapi_MathAPI_scalbln(JNIEnv* env, jclass cls, jdouble x, jlong n) {
    return scalbln(x, n);
}

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
#include "math_api.h"
#include <jni.h>
#include <vcruntime_string.h>
#include <cuda/std/cmath>
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

//double cbrt(double  x)
//double ceil(double  x)
//double copysign(double  x, double  y)
//double cos(double  x)
//double cosh(double  x)
//double cospi(double  x)
//double cyl_bessel_i0(double  x)
//double cyl_bessel_i1(double  x)
//double erf(double  x)
//double erfc(double  x)
//double erfcinv(double  x)
//double erfcx(double  x)
//double erfinv(double  x)
//double exp(double  x)
//double exp10(double  x)
//double exp2(double  x)
//double expm1(double  x)
//double fabs(double  x)
//double fdim(double  x, double  y)
//double floor(double  x)
//double fma(double  x, double  y, double  z)
//double fmax(double, double)
//double fmin(double  x, double  y)
//double fmod(double  x, double  y)
//double frexp(double  x, int* nptr)
//double hypot(double  x, double  y)
//int ilogb(double  x)
//__RETURN_TYPE 	isfinite(double  a)
//__RETURN_TYPE 	isinf(double  a)
//__RETURN_TYPE 	isnan(double  a)
//double j0(double  x)
//double j1(double  x)
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
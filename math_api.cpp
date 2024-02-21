#include "math_api.h"
#include <jni.h>
#include <vcruntime_string.h>
#include <cuda/std/cmath>
#define __CUDA_INTERNAL_COMPILATION__
#include <crt/math_functions.hpp>
#undef __CUDA_INTERNAL_COMPILATION__
#pragma once

#include "Common.h"

#define MAX_ITERATIONS_IKEDA 1000

// #define FLOAT_VERSION
#define DOUBLE_VERSION

void IkedaCPU(uchar* data);

extern "C" void IkedaGPU(uchar* data);

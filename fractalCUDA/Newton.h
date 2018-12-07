#pragma once

#include "Common.h"

#define MAX_ITERATIONS 1000
//#define THRUST_VERSION
#define CUCOMPLEX_VERSION

void NewtonCPU(uchar* data);

extern "C" void NewtonGPU(uchar* data);
#pragma once

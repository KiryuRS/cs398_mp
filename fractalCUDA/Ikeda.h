#pragma once

#include "Common.h"

#define MAX_ITERATIONS_IKEDA 1000

void IkedaCPU(uchar* data);

extern "C" void IkedaGPU(uchar* data);

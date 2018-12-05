#pragma once

#include "Common.h"

#define MAX_ITERATIONS 1000

void NewtonCPU(uchar* data);

extern "C" void NewtonGPU(uchar* data);
#pragma once

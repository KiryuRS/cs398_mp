#pragma once
#include "Common.h"

#define BROWNIAN_ITERATIONS		100000

void BrownianCPU(uchar *data);
void BrownianGPU(uchar* cpuData, uchar** gpuData);
void BrownianClearGPU(uchar **gpuData);
void BrownianGPUKernel(uchar *d_DataIn, uchar *d_DataOut);
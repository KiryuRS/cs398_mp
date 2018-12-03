#pragma once
#include "Common.h"

void BrownianCPU(uchar *data);
void BrownianGPU(uchar* cpuData, uchar** gpuData);
void BrownianGPUKernel(uchar *d_DataIn, uchar *d_DataOut, uint width, uint height);
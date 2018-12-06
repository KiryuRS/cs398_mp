#pragma once
#include "Common.h"
#include <cstdio>

#define SIERPINSKI_DEPTH		5
#define WHITESPACE_PRINT		0
#define HEX_PRINT				1
#define NEWLINE_PRINT			2

void SierpinskiCarpetCPU(uchar *data);
void SierpinskiCarpetGPU(uchar *cpuData, uchar **gpuData);
void SierpinskiCarpetKernel(uchar *d_DataIn, uchar *d_DataOut);
void SierpinskiCarpetClearGPU(uchar **gpuData);
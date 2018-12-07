#pragma once
#include "Common.h"
#include <cstdio>

#define SIERPINSKI_DEPTH		5
#define WHITESPACE_PRINT		1 << 1
#define HEX_PRINT				1 << 2
#define NEWLINE_PRINT			1 << 3

void SierpinskiCarpetCPU(uchar *data);
void SierpinskiCarpetGPU(uchar *cpuData, uchar **gpuData);
void SierpinskiCarpetKernel(uchar *d_DataIn, uint width, uint height);
void SierpinskiCarpetClearGPU(uchar **gpuData);
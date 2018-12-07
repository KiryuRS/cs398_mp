#pragma once


#include "Common.h"


#define iterationM 1000
#define magM 2.5
#define shiftM -PIXELDIM/2.0
#define shiftM2 -PIXELDIM/1.5

struct Mandrelbrot
{
	uchar *ptr1 = nullptr;
	uchar *ptr2 = nullptr;
	uchar *ptr3 = nullptr;
	
	void MandrelbrotCPU(uchar* data);

	void ClearMemory(uchar ** data);
	void MandrelbrotGPU(uchar** gpuOutput);
};


#pragma once


#include "Common.h"


struct Mandrelbrot
{
	uchar *ptr1 = nullptr;
	uchar *ptr2 = nullptr;
	uchar *ptr3 = nullptr;
	
	void MandrelbrotCPU(uchar* data);

	void ClearMemory(uchar ** data);
	void MandrelbrotGPU(uchar** gpuOutput);
};


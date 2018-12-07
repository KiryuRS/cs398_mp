#pragma once

#include "Common.h"
#include <iostream>
struct STriangle
{

	uchar * ptr1 = nullptr;
	uchar * ptr2 = nullptr;
	uchar * ptr3 = nullptr;

	void TriangleCPU(uchar* data);
	
	
	void TriangleGPU(uchar** data);
	
	void ClearMemory(uchar ** data);
	
};
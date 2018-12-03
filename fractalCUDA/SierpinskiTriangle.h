#pragma once

#include "Common.h"


void TriangleCPU(uchar* data);


void TriangleGPU(uchar* CPUin,uchar* data);


void SetData(int x, int y, int value, uchar* data);
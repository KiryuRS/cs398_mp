#pragma once


#include "Common.h"

void MandrelbrotCPU(uchar* data);


 void MandrelbrotGPU(uchar* cpuOutput, uchar** gpuOutput);
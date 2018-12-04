#pragma once

#include "Common.h"

#define iterationBS 100
#define magBS 0.2
#define shiftBS -PIXELDIM/1.6
#define shiftBS2 -PIXELDIM/0.111

void BurningShipCPU(uchar* data);

void BurningShipGPU(uchar** data);


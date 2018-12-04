#pragma once

#include "Common.h"

#define iterationBS 100
#define magBS 0.2
#define shiftBS -PIXELDIM/1.6
#define shiftBS2 -PIXELDIM/0.111



struct BurningShip {
	uchar * ptr1 = nullptr;
	uchar * ptr2 = nullptr;
	uchar * ptr3 = nullptr;



	void BurningShipCPU(uchar* data);

	void BurningShipGPU(uchar** data);

	void clearGPUMemory(uchar** data);


};
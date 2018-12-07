/******************************************************************************/
/*!
@file   BurningShip.h
@par    Purpose: Header file for BurningShip
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author ChengJiang
@par    Email: chengjiang.tham\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#pragma once

#include "Common.h"

#define iterationBS 100
#define magBS 0.2
#define shiftBS -PIXELDIM/1.6
#define shiftBS2 -PIXELDIM/0.111



struct BurningShip {
	uchar * ptr1 = nullptr;



	void BurningShipCPU(uchar* data);

	void BurningShipGPU(uchar** data);

	void clearGPUMemory(uchar** data);


};
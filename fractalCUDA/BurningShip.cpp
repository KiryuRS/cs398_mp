/******************************************************************************/
/*!
@file   BurningShip.cpp
@par    Purpose: Implementation of BurningShip
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author ChengJiang
@par    Email: chengjiang.tham\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#include "BurningShip.h"
#include <iostream>


void BurningShipSetData(int x, int y, int value, uchar* data)
{
	/// to color the pixel
	if (value == 0)
	{
		data[x + PIXELDIM * y] = value; // b
		data[x + PIXELDIM * y + PIXELDIM2] = value; // g
		data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = value; // r
	}
	else
	{
		data[x + PIXELDIM * y] = value; // b
		data[x + PIXELDIM * y + PIXELDIM2] = value; // g
		data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0xBF; // r
	}
}





void BurningShip::BurningShipCPU(uchar* data)
{
	int value = 0;
	for (int j = 0; j < PIXELDIM; ++j) {
		for (int i = 0; i < PIXELDIM; ++i) {
	
			double a = 0.0, b = 0.0, norm2 = 0.0;
			int n;

			/// get the new x and y of zoom and shift to look at the ship
			double x = (double)((i + shiftBS2) *magBS) / PIXELDIM;
			double y = (double)((PIXELDIM - 1 - j + shiftBS) *magBS) / PIXELDIM;

			/// iteration count
			for (n = 0; norm2 < 4.0 && n < iterationBS; ++n) {
				/// calculation
				double c = a*a - b*b + x;
				b = 2.0*std::fabs(a*b) + y;
				a = c;
				norm2 = a*a + b*b;
			}
			/// color value
			int value = (int)(255 * (1 - double(n) / iterationBS));

			/// color the pixel
			BurningShipSetData(i, j, value, data);
		}
	}
	
}


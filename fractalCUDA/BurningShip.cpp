#include "BurningShip.h"
#include <iostream>


void BurningShipSetData(int x, int y, int value, uchar* data)
{
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
	//FractalTreeCPU(data);
	
	int value = 0;
	for (int j = 0; j < PIXELDIM; ++j) {
		for (int i = 0; i < PIXELDIM; ++i) {
	
			double a = 0.0, b = 0.0, norm2 = 0.0;
			int n;
			double x = (double)((i + shiftBS2) *magBS) / PIXELDIM;
			double y = (double)((PIXELDIM - 1 - j + shiftBS) *magBS) / PIXELDIM;
			//std::cout << " start " << x << " " << y << std::endl;
			for (n = 0; norm2 < 4.0 && n < iterationBS; ++n) {
				double c = a*a - b*b + x;
				b = 2.0*a*b;
				b = b <= 0 ? -b : b;
				b += y;
				a = c;
				norm2 = a*a + b*b;
				//std::cout << a << " " << b << std::endl;
				//std::cout << " s " << n << " " << norm2 << std::endl;
			}
			int value = (int)(255 * (1 - double(n) / iterationBS));
			//std::cout << n << " ";
			BurningShipSetData(i, j, value, data);
		}
		//std::cout << std::endl;
		//BitBlt(dc, j, 0, 1, rect.bottom, buffer_dc, j, 0, SRCCOPY);
	}
	
}

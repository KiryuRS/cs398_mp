#include "MandrelBrot.h"
#include <iostream>




void MandrelBrotSetData(int x, int y, int value, uchar* data)
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
		data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = value; // r
	}
}




void Mandrelbrot::MandrelbrotCPU(uchar* data)
{
	int value = 0;
	for (int j = 0; j < PIXELDIM; ++j) 
	{
		for (int i = 0; i < PIXELDIM; ++i)
		{

			double a = 0.0, b = 0.0, norm2 = 0.0;
			int n;
			double x = static_cast<double>(i + shiftM2) *magM / PIXELDIM;
			double y = static_cast<double>(PIXELDIM - 1 - j + shiftM) *magM / PIXELDIM;
			for (n = 0; norm2 < 4.0 && n < iterationM; ++n) 
			{
				double c = a*a - b*b + x;
				b = 2.0*a*b + y;
				a = c;
				norm2 = a*a + b*b;
			
			}
			int value = int(255 * (1 - double(n) / iterationM));
		
			MandrelBrotSetData(i, j, value, data);
		}
		
	}
}


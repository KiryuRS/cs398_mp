#include "BurningShip.h"
#include <iostream>
#define iterationBS 100
#define magBS 0.2
#define shiftBS -PIXELDIM/1.6
#define shiftBS2 -PIXELDIM/0.111

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

void BurningShipGPU(uchar*)
{

}



void BurningShipCPU(uchar* data)
{
	FractalTreeCPU(data);
	/*
	int value = 0;
	for (int j = 0; j < PIXELDIM; ++j) {
		for (int i = 0; i < PIXELDIM; ++i) {
	
			double a = 0.0, b = 0.0, norm2 = 0.0;
			int n;
			double x = static_cast<double>(i + shiftBS2) *magBS / PIXELDIM;
			double y = static_cast<double>(PIXELDIM - 1 - j + shiftBS) *magBS / PIXELDIM;
			//std::cout << " start " << x << " " << y << std::endl;
			for (n = 0; norm2 < 4.0 && n < iterationBS; ++n) {
				double c = a*a - b*b + x;
				b = 2.0*std::fabs(a*b) + y;
				a = c;
				norm2 = a*a + b*b;
				//std::cout << a << " " << b << std::endl;
				//std::cout << " s " << n << " " << norm2 << std::endl;
			}
			int value = int(255 * (1 - double(n) / iterationBS));
			//std::cout << n << " ";
			BurningShipSetData(i, j, value, data);
		}
		//std::cout << std::endl;
		//BitBlt(dc, j, 0, 1, rect.bottom, buffer_dc, j, 0, SRCCOPY);
	}
	*/
}



void SetLineDraw(int x1, int y1, int x2, int y2, uchar* data)
{
	std::cout << x1 << " " << y1 << "              " << x2 << " " << y2 << std::endl;

	int dx = x2 - x1;
	int dy = y2 - y1;

	if (std::abs(dy) > std::abs(dx))
	{
		int absY = std::abs(dy);
		float deltaX = static_cast<float>(dx) / absY;

		float startX = x1- deltaX;

		if (y2 > y1)
		{
			for (int y_i = y1; y_i < y2; y_i++)
			{
				startX += deltaX;

				data[static_cast<int>(startX) + PIXELDIM * y_i] = 0x00; // b
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2] = 0x00; // g
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2 + PIXELDIM2] = 0x00; // r
			}
		}
		else
		{
			for (int y_i = y1; y_i > y2; y_i--)
			{
				startX += deltaX;
				data[static_cast<int>(startX) + PIXELDIM * y_i] = 0x00; // b
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2] = 0x00; // g
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2 + PIXELDIM2] = 0x00; // r
			}
		}

	}
	else
	{
		int absX = std::abs(dx);
		float deltaY = static_cast<float>(dy) / absX;

		float startY = y1 - deltaY;

		if (x2 > x1)
		{
			for (int x_i = x1; x_i < x2; x_i++)
			{
				startY += deltaY;
				data[x_i + PIXELDIM * static_cast<int>(startY)] = 0x00; // b
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2] = 0x00; // g
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2 + PIXELDIM2] = 0x00; // r
			}
		}
		else
		{
			for (int x_i = x1; x_i > x2; x_i--)
			{
				startY += deltaY;
				data[x_i + PIXELDIM * static_cast<int>(startY)] = 0x00; // b
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2] = 0x00; // g
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2 + PIXELDIM2] = 0x00; // r
			}
		}
	}



}


#define r_xcos(x,y) x*0.86602540378f - y*0.5f
#define r_ycos(x,y) y*0.86602540378f + x*0.5f
#define nr_xcos(x,y) x*0.86602540378f + y*0.5f
#define nr_ycos(x,y) y*0.86602540378f - x*0.5f



void RecursionFractalTreeCPU(int locX, int locY, float VecX, float VecY, uchar* data)
{
	float length = std::sqrtf(VecX*VecX + VecY*VecY);
	if (std::sqrtf(VecX*VecX + VecY*VecY) < 3.0f)
		return;
	float vecX_n = VecX / length;
	float vecY_n = VecY / length;

	float d1_x = r_xcos(vecX_n, vecY_n);
	float d1_y = r_ycos(vecX_n, vecY_n);




	float d2_x = nr_xcos(vecX_n, vecY_n);
	float d2_y = nr_ycos(vecX_n, vecY_n);

	//std::cout << d1_x << " " << d1_y << " h " << d2_x << " " << d2_y << std::endl;

	length *= 3.0f / 4.0f;

	SetLineDraw(locX, locY, locX + d1_x * length,locY + d1_y * length, data);
	SetLineDraw(locX, locY, locX + d2_x * length,locY + d2_y * length, data);

	RecursionFractalTreeCPU(locX + d1_x * length, locY + d1_y * length, d1_x * length, d1_y * length, data);
	RecursionFractalTreeCPU(locX + d2_x * length, locY + d2_y * length, d2_x * length, d2_y * length, data);

}


void FractalTreeCPU(uchar* data)
{

	float x0 = PIXELDIM / 2.0f;
	float y0 = 200.f;

	float x1 = PIXELDIM / 2.0f;
	float y1 = 250.f;

	float dx = x1 - x0;
	float dy = y1 - y0;

	SetLineDraw(x0, y0, x1, y1, data);


	RecursionFractalTreeCPU(x1, y1, dx, dy, data);
}

void FractalTreeGPU(uchar* data)
{






}
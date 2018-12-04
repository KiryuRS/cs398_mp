#include "FractalTree.h"
#include <iostream>



void SetLineDraw(int x1, int y1, int x2, int y2, uchar* data, uint counter)
{
	//std::cout << x1 << " " << y1 << "              " << x2 << " " << y2 << std::endl;

	int dx = x2 - x1;
	int dy = y2 - y1;

	counter *= 12;

	if (std::abs(dy) > std::abs(dx))
	{
		int absY = std::abs(dy);
		float deltaX = static_cast<float>(dx) / absY;

		float startX = x1 - deltaX;

		if (y2 > y1)
		{
			for (int y_i = y1; y_i < y2; y_i++)
			{
				startX += deltaX;

				data[static_cast<int>(startX) + PIXELDIM * y_i] = 0x00; // b
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2] = counter; // g
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2 + PIXELDIM2] = 255 - counter; // r
			}
		}
		else
		{
			for (int y_i = y1; y_i > y2; y_i--)
			{
				startX += deltaX;
				data[static_cast<int>(startX) + PIXELDIM * y_i] = 0x00; // b
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2] = counter; // g
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2 + PIXELDIM2] = 255 - counter; // r
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
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2] = counter; // g
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2 + PIXELDIM2] = 255 - counter; // r
			}
		}
		else
		{
			for (int x_i = x1; x_i > x2; x_i--)
			{
				startY += deltaY;
				data[x_i + PIXELDIM * static_cast<int>(startY)] = 0x00; // b
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2] = counter; // g
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2 + PIXELDIM2] = 255 - counter; // r
			}
		}
	}



}


#define r_xcos(x,y) x*0.86602540378f - y*0.5f
#define r_ycos(x,y) y*0.86602540378f + x*0.5f
#define nr_xcos(x,y) x*0.86602540378f + y*0.5f
#define nr_ycos(x,y) y*0.86602540378f - x*0.5f



void RecursionFractalTreeCPU(int locX, int locY, float VecX, float VecY, uchar* data, uint counter)
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

	length *= 13.0f / 16.0f;

	SetLineDraw(locX, locY, locX + d1_x * length, locY + d1_y * length, data, counter + 1);
	SetLineDraw(locX, locY, locX + d2_x * length, locY + d2_y * length, data, counter + 1);

	RecursionFractalTreeCPU(locX + d1_x * length, locY + d1_y * length, d1_x * length, d1_y * length, data, counter + 1);
	RecursionFractalTreeCPU(locX + d2_x * length, locY + d2_y * length, d2_x * length, d2_y * length, data, counter + 1);
	//std::cout << counter + 1 << std::endl;
}


void FractalTreeCPU(uchar* data)
{

	float x0 = PIXELDIM / 2.0f;
	float y0 = 50.f;

	float x1 = PIXELDIM / 2.0f;
	float y1 = 120.f;

	float dx = x1 - x0;
	float dy = y1 - y0;

	uint counter = 0;

	SetLineDraw(x0, y0, x1, y1, data, counter);


	RecursionFractalTreeCPU(x1, y1, dx, dy, data, counter);
}


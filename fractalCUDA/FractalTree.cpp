#include "FractalTree.h"
#include <iostream>



void SetLineDraw(int x1, int y1, int x2, int y2, uchar* data, uint counter)
{
	
#ifdef fTdraw

	/// draw of line
	int dx = x2 - x1;
	int dy = y2 - y1;
	counter *= 5;
	//counter = counter > 255 ? 255 : 0;

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


#endif
}




void RecursionFractalTreeCPU(int locX, int locY, float VecX, float VecY, uchar* data, uint counter)
{
	/// check length
	float length = std::sqrtf(VecX*VecX + VecY*VecY);
	if (std::sqrtf(VecX*VecX + VecY*VecY) < lim)
		return;

	/// get the new vector
	float vecX_n = VecX / length;
	float vecY_n = VecY / length;

	float d1_x = r_xcos(vecX_n, vecY_n);
	float d1_y = r_ycos(vecX_n, vecY_n);




	float d2_x = nr_xcos(vecX_n, vecY_n);
	float d2_y = nr_ycos(vecX_n, vecY_n);

	/// new length
	length *= per;

	///draw line
	SetLineDraw(locX, locY, locX + d1_x * length, locY + d1_y * length, data, counter + 1);
	SetLineDraw(locX, locY, locX + d2_x * length, locY + d2_y * length, data, counter + 1);

	/// recursion
	RecursionFractalTreeCPU(locX + d1_x * length, locY + d1_y * length, d1_x * length, d1_y * length, data, counter + 1);
	RecursionFractalTreeCPU(locX + d2_x * length, locY + d2_y * length, d2_x * length, d2_y * length, data, counter + 1);
}


void FractalTree::FractalTreeCPU(uchar* data)
{
	/// set up for the first line
	float x0 = PIXELDIM / 2.0f;
	float y0 = sY;

	float x1 = PIXELDIM / 2.0f;
	float y1 = eY;

	float dx = x1 - x0;
	float dy = y1 - y0;

	uint counter = 0;
	/// draw the first line
	SetLineDraw(x0, y0, x1, y1, data, counter);

	/// recurse
	RecursionFractalTreeCPU(x1, y1, dx, dy, data, counter);
}


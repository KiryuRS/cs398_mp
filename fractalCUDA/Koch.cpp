#include "Koch.h"

#define PI 3.14159265359
#define iterationBS 64 
#define magBS 2.5
#define shiftBS -PIXELDIM/2.0
#define shiftBS2 -PIXELDIM/1.5



void KochSetData(int x, int y, int value, uchar* data)
{
	if (value == 0)
	{
		data[x + PIXELDIM * y] = 0xad; // b
		data[x + PIXELDIM * y + PIXELDIM2] = 0x16; // g
		data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0x89; // r
	}
	else
	{
		data[x + PIXELDIM * y] = value; // b
		data[x + PIXELDIM * y + PIXELDIM2] = value; // g
		data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0xBF; // r
	}
}

void DrawLine(int x1, int y1, int x2, int y2, uchar * data)
{
	int dx = x2 - x1;
	int dy = y2 - y1;

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


void KochGPU(uchar * data)
{
}


void KochCPU(uchar * data)
{
	int x1 = PIXELDIM/2, y1 = 200, x2 = PIXELDIM/2, y2 = 200;
	for (int j = 0; j < PIXELDIM; ++j)
	{
		for (int i = 0; i < PIXELDIM; ++i)
		{
			
			KochRecursive(x1, y1, x2, y2, iterationBS);
			int value = int(255 * (1 - iterationBS) / iterationBS);
			KochSetData(i, j, value, data);

		}
	
	}
}

void KochRecursive(int x1, int y1, int x2, int y2, int itr)
{
	float angle = 60 * PI / 180;
	int x3 = (2 * x1 + x2) / 3;
	int y3 = (2 * y1 + y2) / 3;

	int x4 = (x1 + 2 * x2) / 3;
	int y4 = (y1 + 2 * y2) / 3;

	int x = x3 + (x4 - x3)*cos(angle) + (y4 - y3)*sin(angle);
	int y = y3 + (x4 - x3)*sin(angle) + (y4 - y3)*sin(angle);

	if (itr > 0)
	{
		KochRecursive(x1, y1, x3, y3, itr - 1);
		KochRecursive(x3, y3, x, y, itr - 1);
		KochRecursive(x, y, x4, y4, itr - 1);
		KochRecursive(x4, y4, x2, y2, itr - 1);
	}
	else
	{
		//line(x1, y1, x3, y3);
		//line(x3, y3, x, y);
		//line(x, y, x4, y4);
		//line(x4, y4, x2, y2);
	}
}

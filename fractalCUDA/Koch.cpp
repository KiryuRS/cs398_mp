#include "Koch.h"

#define PI 3.14159265359
#define iterationBS 64 
#define magBS 2.5
#define shiftBS -PIXELDIM/2.0
#define shiftBS2 -PIXELDIM/1.5





void SetLine(int x1, int y1, int x2, int y2, uchar * data)
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
	float x0 = 200.f;
	float y0 = PIXELDIM / 2.0f;

	float x1 = 250.f;
	float y1 = PIXELDIM / 2.0f;

	float dx = x1 - x0;
	float dy = y1 - y0;

	//First Line
	SetLine(x0, y0, x1, y1, data);
	//SetLine(300.f, y0, 350.f, y1, data);
	float direction = 60;
	KochRecursive(x1, y1, dx,dy, direction, 5, data);
    
	//KochRecursive(0.0, 0.004, 5, data);
	//KochRecursive(-120.0, 0.004, 5, data);
	//KochRecursive(120.0, 0.004, 5, data);
	
	
}

#define r_xcos(x,y) x*0.86602540378f - y*0.5f
#define r_ycos(x,y) y*0.86602540378f + x*0.5f
#define nr_xcos(x,y) x*0.86602540378f + y*0.5f
#define nr_ycos(x,y) y*0.86602540378f - x*0.5f

void KochRecursive(float x1, float y1,float x2, float y2, int direction, int itr, uchar* data)
{
	float length = std::sqrtf(x2*x2 + y2*y2);
	if (length < 3.0f)
		return;
	float vecX_n = x2 / length;
	float vecY_n = y2 / length;

	float d1_x = r_xcos(vecX_n, vecY_n);
	float d1_y = r_ycos(vecX_n, vecY_n);

	float d2_x = nr_xcos(vecX_n, vecY_n);
	float d2_y = nr_ycos(vecX_n, vecY_n);

	//double dirRad = 0.0174533 * direction;
//	float length = x2 - x1;
	//float newX = x2 + length * cos(dirRad);
	//float newY = y2 + length * sin(dirRad);

	//draw the four parts of the side _/\_ 
	//KochRecursive(direction, length, itr, data);
	//direction += 60.0;
	//KochRecursive(direction, length, itr, data);
	//direction -= 120.0;
	//KochRecursive(direction, length, itr, data);
	//direction += 60.0;
	//KochRecursive(direction, length, itr, data);
		
	
	float dx = x2 - x1;
	float dy = y2 - y1;
	
	//int x = x2 + (x2 - x1)*cos(direction) + (y2 - y1)*sin(direction);
	//int y = y2 - (x2 - x1)*sin(direction) + (y2 - y1)*sin(direction);
	
	if (itr > 0)
	{
		itr--;
		SetLine(x1, y1, x1+ d1_x * length, y1 + d1_y * length, data);
		SetLine(x1 + d1_x * length, y1 + d1_y * length, 
                x1 + d1_x * length + d2_x * length, 
			    y1 + d1_y * length + d2_y * length, data);

		float nextX = x1 + d1_x * length + d2_x * length;
		float nextY = y1 + d1_y * length + d2_y * length;
		SetLine(nextX, nextY, nextX+d1_x*length, nextY, data);
		direction += 60;
		float futureX = nextX + d1_x * length ;
		float futureY = nextY + d1_y*length;
		
		//std::cout << "x:  " << newX << "  y2 : " << newY << std::endl;
		//KochRecursive(nextX + d1_x*length, nextY, futureX,futureY, direction, itr, data);
		//			y1 + d1_y * length + d2_y * length, x1+dx, y1+dy,direction, itr ,data);
		
		//KochRecursive(x3, y3, x, y, direction, itr ,data);
		//SetLine(x3, y3, x, y, data);
		//direction -= 120;
		//KochRecursive(x, y, x4, y4, direction, itr ,data);
		//SetLine(x, y, x4, y4, data);
		//direction += 60;
		//KochRecursive(x4, y4, x2, y2, direction, itr ,data);
		//SetLine(x4, y4, x2, y2, data);
	}
	
}

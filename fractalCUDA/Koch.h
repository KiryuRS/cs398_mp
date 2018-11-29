#pragma once
#include "Common.h"
#include <list>
#define PI 3.14159265359

#define iterationBS 64 
#define magBS 2.5
#define shiftBS -PIXELDIM/2.0
#define shiftBS2 -PIXELDIM/1.5

using namespace std;


class Line
{
public:
	float x, y, len, ang;

	Line(float x,float y, float len, float ang):x(x), y(y),len(len),ang(ang)
	{}

	float SecondXCoordinate()
	{
		return x + cos(ang* (PI / 180))* len;
	}
	float SecondYCoordinate()
	{
		return y + sin(ang * (PI / 180))*len;
	}
	void draw(int x, int y, int value, uchar* data)
	{

		if (value == 0)
		{
			data[x + PIXELDIM * y] = 0x2f; // b
			data[x + PIXELDIM * y + PIXELDIM2] = 0xff; // g
			data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0xad; // r
		}
		else
		{
			data[x + PIXELDIM * y] = value; // b
			data[x + PIXELDIM * y + PIXELDIM2] = value; // g
			data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0xBF; // r
		}
	}
	
};

void KochCPU(list<Line*> &lines);
void RunKoch(uchar* data);
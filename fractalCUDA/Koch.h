#pragma once
#include "Common.h"



void KochGPU(uchar* data);
	 
void KochCPU(uchar* data);

void KochRecursive(float newX, float newY, float nextX, float nextY, int direction, int itr, uchar* data);


void SetLine(int x1, int y1, int x2, int y2, uchar* data);



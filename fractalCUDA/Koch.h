#pragma once
#include "Common.h"



void KochGPU(uchar* data);
	 
void KochCPU(uchar* data);

void KochRecursive(int x1, int y1, int x2, int y2, int itr);

void KochSetData(int x, int y, int value, uchar* data);


void DrawLine(int x1, int y1, int x2, int y2, uchar* data);
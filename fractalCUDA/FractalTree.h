/******************************************************************************/
/*!
@file   FractalTree.h
@par    Purpose: Header file for FractalTree
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author ChengJiang
@par    Email: chengjiang.tham\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#pragma once

#include "Common.h"



#define r_xcos(x,y) x*0.86602540378f - y*0.5f
#define r_ycos(x,y) y*0.86602540378f + x*0.5f
#define nr_xcos(x,y) x*0.86602540378f + y*0.5f
#define nr_ycos(x,y) y*0.86602540378f - x*0.5f

#define sY 50.0f
#define eY 120.0f
#define per 13.0f/16.0f
#define lim 3.0f
#define fTsingleBlock 1024
#define fTsharedMem 4096
#define fTdraw
struct FractalTree
{
	uchar * ptr1 = nullptr;
	float * fptr1 = nullptr;

	float * fptr2 = nullptr;

	void FractalTreeCPU(uchar* data);

	void FractalTreeGPU(uchar** data);


	void clearGPUMemory(uchar** data);
};
/******************************************************************************/
/*!
@file   SierpinskiTriangle.h
@par    Purpose: Header file for SierpinskiTriangle
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author YongKiat
@par    Email: yongkiat.ong\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#pragma once

#include "Common.h"
#include <iostream>
struct STriangle
{

	uchar * ptr1 = nullptr;
	uchar * ptr2 = nullptr;
	uchar * ptr3 = nullptr;

	void TriangleCPU(uchar* data);
	
	
	void TriangleGPU(uchar** data);
	
	void ClearMemory(uchar ** data);
	
};
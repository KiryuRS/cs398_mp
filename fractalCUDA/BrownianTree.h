/******************************************************************************/
/*!
@file   BrownianTree.h
@par    Purpose: Header file for BrownianTree
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author Kenneth
@par    Email: t.weigangkenneth\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#pragma once
#include "Common.h"

#define BROWNIAN_ITERATIONS		100000

void BrownianCPU(uchar *data);
void BrownianGPU(uchar* cpuData, uchar** gpuData);
void BrownianClearGPU(uchar **gpuData);
void BrownianGPUKernel(uchar *d_DataIn, uchar *d_DataOut);
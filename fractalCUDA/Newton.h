/******************************************************************************/
/*!
@file   Newton.h
@par    Purpose: Header file for Newton
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author Alvin
@par    Email: alvin.tan\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#pragma once

#include "Common.h"

#define MAX_ITERATIONS 1000
//#define THRUST_VERSION
#define CUCOMPLEX_VERSION

void NewtonCPU(uchar* data);

extern "C" void NewtonGPU(uchar* data);
#pragma once

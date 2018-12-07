/******************************************************************************/
/*!
@file   Ikeda.h
@par    Purpose: Header file for Ikeda
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author Alvin
@par    Email: alvin.tan\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#pragma once

#include "Common.h"

#define MAX_ITERATIONS_IKEDA 1000

#define FLOAT_VERSION
//#define DOUBLE_VERSION

void IkedaCPU(uchar* data);

extern "C" void IkedaGPU(uchar* data);

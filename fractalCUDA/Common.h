/******************************************************************************/
/*!
@file   Common.h
@par    Purpose: Header file for all common includes
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author Kenneth, Alvin, Yong Kiat, Cheng Jiang
@par    Email:	t.weigangkenneth\@digipen.edu, alvin.tan\@digipen.edu,
yongkiat.ong\@digipen.edu, chengjiang.tham\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#pragma once

///////////////////////////////////////////////////////////////////////////
// Common Types, Defines and Includes
///////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdint>
#include <fstream>
#include <thread>
#include <chrono>
#include <memory>
#include "bmp.h"

typedef unsigned int				uint;
typedef unsigned char				uchar;
typedef unsigned short				ushort;
typedef unsigned long				ul;
typedef long long					ll;
typedef unsigned long long 			ull;

#ifndef MAX
#define MAX(a,b) \
((a > b) ? a : b)
#endif

#ifndef MIN
#define MIN(a,b) \
((a < b) ? a : b)
#endif

#define UINT_BITS					32
#define BLOCK_SIZE					16
#define PIXELDIM		            512
static constexpr size_t PIXELDIM2 = PIXELDIM * PIXELDIM;
static constexpr size_t PIXELDIM3 = PIXELDIM * PIXELDIM * 3;

#define EPSILON						10e-9

///////////////////////////////////////////////////////////////////////////
// CUDA Includes
///////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_functions.h"
#include <helper_cuda.h>
#include <helper_functions.h>


///////////////////////////////////////////////////////////////////////////
// Helper Function Calls and Misc.
///////////////////////////////////////////////////////////////////////////
template <typename InputIter, typename OutputIter>
OutputIter MyCopy(InputIter begin, InputIter end, OutputIter start)
{
	while (begin != end)
		*start++ = *begin++;
	return start;
}

///////////////////////////////////////////////////////////////////////////
// Fractals Includes
///////////////////////////////////////////////////////////////////////////
#include "Henon.h"
#include "Newton.h"
#include "Ikeda.h"
#include "BurningShip.h"
#include "MandrelBrot.h"
#include "SierpinskiTriangle.h"
#include "FractalTree.h"
#include "SierpinskiCarpet.h"
#include "BrownianTree.h"

///////////////////////////////////////////////////////////////////////////
// User Data Types
///////////////////////////////////////////////////////////////////////////

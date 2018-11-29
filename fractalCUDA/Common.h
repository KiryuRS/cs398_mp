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
#include "bmp.h"

typedef unsigned int				uint;
typedef unsigned char				uchar;
typedef unsigned short				ushort;
typedef unsigned long				ulong;
typedef long long					llong;

#ifndef MAX
#define MAX(a,b) \
((a > b) ? a : b)
#endif

#ifndef MIN
#define MIN(a,b) \
((a < b) ? a : b)
#endif

#define UINT_BITS					32
#define PIXELDIM		            512
static constexpr size_t PIXELDIM2 = PIXELDIM * PIXELDIM;
static constexpr size_t PIXELDIM3 = PIXELDIM * PIXELDIM * 3;

///////////////////////////////////////////////////////////////////////////
// CUDA Includes
///////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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
#include "henon.h"
#include "BurningShip.h"
#include "MandrelBrot.h"
#include "Koch.h"
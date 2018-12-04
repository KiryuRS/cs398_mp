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
#include "SierpinskiTriangle.h"
#include "FractalTree.h"

///////////////////////////////////////////////////////////////////////////
// User Data Types
///////////////////////////////////////////////////////////////////////////
template <typename T = uchar, typename Allocator = std::allocator<T>>
class RTTW
{
	T* container;
	size_t size_;
	Allocator alloc;

public:
	RTTW() : container{ nullptr }, size_{ 0 }, alloc{ }
	{ }

	RTTW(const RTTW& rhs) : alloc{ rhs.alloc }, size_{ rhs.size_ }, container{ alloc.allocate(size_) }
	{
		MyCopy(rhs.container, rhs.container + size_, container);
	}

	RTTW(size_t size) : alloc{ }, size_{ size }, container{ alloc.allocate(size) }
	{ }

	RTTW(T* data, size_t size) : alloc{ }, size_{ size }, container{ alloc.allocate(size) }
	{
		MyCopy(data, data + size, container);
	}

	RTTW(RTTW&&) = delete;

	RTTW& operator=(const RTTW& rhs)
	{
		T* tmp = alloc.allocate(rhs.size_);
		MyCopy(rhs.container, rhs.container + size_, tmp);
		alloc.deallocate(container, size_);
		alloc = rhs.alloc;
		container = tmp;
		size_ = rhs.size_;
		return *this;
	}

	RTTW& operator=(RTTW&) = delete;

	~RTTW()
	{
		DeallocateMemory();
	}

	void AllocateMemory(size_t sz)
	{
		if (container)
			alloc.deallocate(container, size_);
		alloc.allocate(sz);
		size_ = sz;
	}

	void DeallocateMemory()
	{
		if (container)
			alloc.deallocate(container, size_);
	}

	size_t size(void) const
	{
		return size_;
	}

	bool empty(void) const
	{
		return !size_;
	}

	T* get() const
	{
		return container;
	}

	T* begin() const
	{
		return container;
	}

	T* end() const
	{
		return container + size_;
	}
};
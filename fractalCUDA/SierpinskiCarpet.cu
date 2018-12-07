/******************************************************************************/
/*!
@file   SierpinskiCarpet.cu
@par    Purpose: Implementation of SierpinskiCarpet CUDA kernel
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author Kenneth
@par    Email: t.weigangkenneth\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#include "Common.h"

__global__ void siecarpCalculation(uchar *d_DataIn, uint width, uint height)
{
	const uint dim = height - 2;
	const uint dimW = width - 1;

	uint ty = threadIdx.y;
	uint tx = threadIdx.x;
	uint y = blockIdx.y * blockDim.y + ty;
	uint x = blockIdx.x * blockDim.x + tx;

	// Check if its within boundaries
	if (y >= dim)
		return;

	if (x >= dimW)
	{
		d_DataIn[y * width + dim] = NEWLINE_PRINT;
		return;
	}

	uint d = 0;
	for (d = dim / SIERPINSKI_DEPTH; d; d /= SIERPINSKI_DEPTH)
		if ((y % (d * SIERPINSKI_DEPTH)) / d == 1 && (x % (d * SIERPINSKI_DEPTH)) / d == 1)
			break;

	// Checking for the value of d and append to the data
	d_DataIn[y * width + x] = d ? WHITESPACE_PRINT : HEX_PRINT;
}

void SierpinskiCarpetKernel(uchar *d_DataIn, uint width, uint height)
{
	// Setup
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(ceil((float)width / BLOCK_SIZE), ceil((float)height / BLOCK_SIZE));

	// Calling the function
	siecarpCalculation<<<dimGrid, dimBlock>>>(d_DataIn, width, height);
}

void SierpinskiCarpetFree()
{
	cudaDeviceReset();
}

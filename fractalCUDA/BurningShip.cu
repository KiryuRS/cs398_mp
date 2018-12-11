/******************************************************************************/
/*!
@file   BurningShip.cu
@par    Purpose: Implementation of BurningShip CUDA kernel
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author ChengJiang
@par    Email: chengjiang.tham\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#include "Common.h"
#include <cuda_fp16.h>

#define shipBlock_size 16

#define BurningshipDefault			// BurningshipDefault, BurningshipUnified, BurningshipPinned
#define BurnHalf					// BurnDefault,BurnInstrinic, BurnHalf


__global__ void BurningShipDefaultCu(uchar *d_DataOut, uint limit)
{
	/// get position
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;

	if (tx >= PIXELDIM || ty >= PIXELDIM)
		return;

	double a = 0.0;
	double b = 0.0;
	double norm2 = 0.0;
	int n;
	/// the shifting of the image to look at the ships
	double x = (double)((tx + shiftBS2) *magBS) / PIXELDIM;
	double y = (double)((PIXELDIM - 1 - ty + shiftBS) *magBS) / PIXELDIM;
	double iter = 1;

	/// iteration fns
	for (n = 0; norm2 < 4.0 && n < iterationBS; ++n) {
		/// calculation
		double c = a*a - b*b + x;
		b = 2.0*a*b;
		b = b < 0 ? -b : b;
		b += y;
		a = c;
		norm2 = a*a + b*b;
	}

	/// to get the color value
	iter -= double(n) / iterationBS;
	int value = (int)(255 * iter);
	
	
	/// color the pixel
	if (value == 0)
	{
		d_DataOut[tx + PIXELDIM * ty] = value; // b
		d_DataOut[tx + PIXELDIM * ty + PIXELDIM2] = value; // g
		d_DataOut[tx + PIXELDIM * ty + PIXELDIM2 + PIXELDIM2] = value; // r
	}
	else
	{
		d_DataOut[tx + PIXELDIM * ty] = value; // b
		d_DataOut[tx + PIXELDIM * ty + PIXELDIM2] = value; // g
		d_DataOut[tx + PIXELDIM * ty + PIXELDIM2 + PIXELDIM2] = 0xBF; // r
	}
}


/// Instrinics set up


__forceinline__ __device__ double fastAbsD(double in)
{
	return fabs(in);
}


__forceinline__ __device__ double getC(double a, double b, double c)
{
	return __dsub_rd(__fma_rd(a,a,c) ,__dmul_rd(b, b));
}

__forceinline__ __device__ double getb(double a, double b, double c)
{
	return __fma_rd(2.0, fabs(__dmul_rd(a, b)), c);
}

__forceinline__ __device__ double getNorm(double a, double b)
{
	return __fma_rd(a, a, __dmul_rd(b, b));
}


__forceinline__ __device__ double simpleMulti(double a, double b)
{
	return __dmul_rd(a, b);
}

__forceinline__ __device__ double getIter(double n, double total)
{
	return __dsub_rd(1.0, __ddiv_rd(n, total));
}

__forceinline__ __device__ double valueGet(double n, double total)
{
	return __double2int_rd(__dmul_rd(255.0, __dsub_rd(1.0, __ddiv_rd(n, total))));
}

__forceinline__ __device__ double getX(double a, double b, double c, double d)
{
	return __ddiv_rd(__dmul_rd(__dadd_rd(a, b), c), d);
}

__forceinline__ __device__ double getY(double a, double b, double c, double d, double e)
{
	return __ddiv_rd(__dmul_rd(__dsub_rd(__dadd_rd(a, c), ++b), d), e);
}

__forceinline__ __device__ int getLoc(double a, double b, double c)
{
	return __double2int_rd(fma(b, c, a));


}

#pragma region HALF_PRECISION

__forceinline__ __device__ half halfAbs(const half& in)
{
	// return __hlt(in, 0.f) ? __hneg(in) : in;
	return fabs(in);
}

__forceinline__ __device__ half halfGetC(const half& a, const half& b, const half& c)
{
	return __hsub(__hfma(a, a, c), __hmul(b, b));
  //return half{ __dsub_rd(__fma_rd(a, a, c), __dmul_rd(b, b)) };
}

__forceinline__ __device__ half halfGetB(const half& a, const half& b, const half& c)
{
	return __hfma(2.0f, fabs(__hmul(a, b)), c);
  //return half{ __fma_rd(2.0, fabs(__dmul_rd(a, b)), c) };
}

__forceinline__ __device__ half halfGetNorm(const half& a, const half& b)
{
	return __hfma(a, a, __hmul(b, b));
  //return half{ __fma_rd(a, a, __dmul_rd(b, b)) };
}

__forceinline__ __device__ half halfGetIter(const half& n, const half& total)
{
	return __hsub(half{ 1.0f }, __hdiv(n, total));
  //return half{ __dsub_rd(1.0, __ddiv_rd(n, total)) };
}

__forceinline__ __device__ int halfValueGet(const half& n, const half& total)
{
	return __half2int_rd(__hmul(half{ 255.0f }, __hsub(half{ 1.0f }, __hdiv(n, total))));
}

__forceinline__ __device__ half halfGetX(const half& a, const half& b, const half& c, const half& d)
{
	return __hdiv(__hmul(__hadd(a, b), c), d);
  //return half{ __ddiv_rd(__dmul_rd(__dadd_rd(a, b), c), d) };
}

__forceinline__ __device__ half halfGetY(const half& a, const half& b, const half& c, const half& d, const half& e)
{
	return __hdiv(__hmul(__hsub(__hadd(a, c), __hadd(b, half{ 1.0f })), d), e);
  //return half{ __ddiv_rd(__dmul_rd(__dsub_rd(__dadd_rd(a, c), ((float)b)+1.0f), d), e) };
}
//
//__forceinline__ __device__ half halfGetLoc(double a, double b, double c)
//{
//	return __half2int_rd(fma(b, c, a));
//}

__global__ void BurningShipHalfPrecisionCu(uchar *d_DataOut, uint limit)
{
	// get the location
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;

	if (tx >= PIXELDIM || ty >= PIXELDIM)
		return;

	half a{ 0.0f };
	half b{ 0.0f };
	half norm2{ 0.0f };
	int n;

	// image zoom and shift to look at ship
  double x = getX(tx, shiftBS2, magBS, PIXELDIM);
  double y = getY(PIXELDIM, ty, shiftBS, magBS, PIXELDIM);

	// iterative fns (half precision)
  half four{ 4.0f };
	for (n = 0; __hlt(norm2, four) && n < iterationBS; ++n)
	{
		// calculation using instrinics
		half c = halfGetC(a, b, x);
		b = halfGetB(a, b, y);
		a = c;
		norm2 = halfGetNorm(a, b);
	}

	// get the value to color
	int value = valueGet(n, iterationBS);
	int loc = getLoc(tx, PIXELDIM, ty);

	// color pixel
	if (loc < PIXELDIM2)
	{
		if (value == 0)
		{
			d_DataOut[loc] = value; // b
			d_DataOut[loc + PIXELDIM2] = value; // g
			d_DataOut[loc + PIXELDIM2 + PIXELDIM2] = value; // r
		}
		else
		{
			d_DataOut[loc] = value; // b
			d_DataOut[loc + PIXELDIM2] = value; // g
			d_DataOut[loc + PIXELDIM2 + PIXELDIM2] = 0xBF; // r
		}
	}
}

#pragma endregion


__global__ void BurningShipIntrinsicsCu(uchar *d_DataOut, uint limit)
{
	/// get the location
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;

	if (tx >= PIXELDIM || ty >= PIXELDIM)
		return;

	double a = 0.0;
	double b = 0.0;
	double norm2 = 0.0;
	int n;
	/// image zoom and shift to look at ship
	double x = getX(tx, shiftBS2, magBS, PIXELDIM);
	double y = getY(PIXELDIM, ty, shiftBS, magBS, PIXELDIM);

	/// iterative fns 
	for (n = 0; norm2 < 4.0 && n < iterationBS; ++n) {
		/// calculation using instrinics
		double c = getC(a,b,x);
		b = getb(a, b, y);
		a = c;
		norm2 = getNorm(a,b);
	}

	/// get the value to color
	int value = valueGet(n, iterationBS);
	int loc = getLoc(tx, PIXELDIM, ty);

	/// color pixel
	if (value == 0)
	{
		d_DataOut[loc] = value; // b
		d_DataOut[loc + PIXELDIM2] = value; // g
		d_DataOut[loc + PIXELDIM2 + PIXELDIM2] = value; // r
	}
	else
	{
		d_DataOut[loc] = value; // b
		d_DataOut[loc + PIXELDIM2] = value; // g
		d_DataOut[loc + PIXELDIM2 + PIXELDIM2] = 0xBF; // r
	}
}





//-fmad=false command line to fix precision issue due to default optimize by compiler

void BurningShip::BurningShipGPU(uchar** data)
{
#ifdef BurningshipDefault


	dim3 DimBlock(shipBlock_size, shipBlock_size, 1);
	dim3 DimGrid(ceil(((float)PIXELDIM) / shipBlock_size), ceil(((float)PIXELDIM) / shipBlock_size), 1);


	checkCudaErrors(cudaMalloc((void **)&ptr1, PIXELDIM3 * sizeof(uchar)));
#ifdef BurnDefault
	BurningShipDefaultCu << <DimGrid, DimBlock >> > (ptr1, PIXELDIM);
#elif defined BurnInstrinic
	BurningShipIntrinsicsCu << <DimGrid, DimBlock >> > (ptr1, PIXELDIM);
#elif defined BurnHalf
	BurningShipHalfPrecisionCu << <DimGrid, DimBlock >> > (ptr1, PIXELDIM);
	checkCudaErrors(cudaGetLastError());
#endif

	cudaDeviceSynchronize();

	*data = (uchar *)malloc(PIXELDIM3 * sizeof(uchar));
	checkCudaErrors(cudaMemcpy(*data, ptr1, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));


#elif defined BurningshipUnified

	dim3 DimBlock(shipBlock_size, shipBlock_size, 1);
	dim3 DimGrid(ceil(((float)PIXELDIM) / shipBlock_size), ceil(((float)PIXELDIM) / shipBlock_size), 1);

	checkCudaErrors(cudaMallocManaged((void **)data, PIXELDIM3 * sizeof(uchar)));
#ifdef BurnDefault
	BurningShipDefaultCu << <DimGrid, DimBlock >> > (*data, PIXELDIM);
#elif defined BurnInstrinic
	BurningShipIntrinsicsCu << <DimGrid, DimBlock >> > (*data, PIXELDIM);
#elif defined BurnHalf
	BurningShipHalfPrecisionCu << <DimGrid, DimBlock >> > (*data, PIXELDIM);
#endif

	cudaDeviceSynchronize();

#elif defined BurningshipPinned


	dim3 DimBlock(shipBlock_size, shipBlock_size, 1);
	dim3 DimGrid(ceil(((float)PIXELDIM) / shipBlock_size), ceil(((float)PIXELDIM) / shipBlock_size), 1);


	checkCudaErrors(cudaMalloc((void **)&ptr1, PIXELDIM3 * sizeof(uchar)));
#ifdef BurnDefault
	BurningShipDefaultCu << <DimGrid, DimBlock >> > (ptr1, PIXELDIM);
#elif defined BurnInstrinic
	BurningShipIntrinsicsCu << <DimGrid, DimBlock >> > (ptr1, PIXELDIM);
#elif defined BurnHalf
	BurningShipHalfPrecisionCu << <DimGrid, DimBlock >> > (ptr1, PIXELDIM);
#endif

	cudaDeviceSynchronize();
	checkCudaErrors(cudaHostAlloc((void **)data, PIXELDIM3 * sizeof(uchar), cudaHostAllocDefault));
	checkCudaErrors(cudaMemcpy(*data, ptr1, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));


#endif




}

void BurningShip::clearGPUMemory(uchar** data)
{
#ifdef BurningshipDefault

	checkCudaErrors(cudaFree(ptr1));
	free(*data);

#elif defined BurningshipUnified
	checkCudaErrors(cudaFree(*data));


#elif defined BurningshipPinned
	checkCudaErrors(cudaFreeHost(*data));
	checkCudaErrors(cudaFree(ptr1));
#endif
}
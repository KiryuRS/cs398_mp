/******************************************************************************/
/*!
@file   Ikeda.cu
@par    Purpose: Implementation of Ikeda CUDA kernel
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author Alvin
@par    Email: alvin.tan\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#include "Common.h"

#if defined (DOUBLE_VERSION)
__constant__ double min_x = -0.5, min_y = -2.5;
__constant__ double max_x = 6.5, max_y = 6.5;
__constant__ double u = 0.918;

__device__ __forceinline__ double t_next(double x, double y)
{
  return 0.4 - 6.0 / (1.0 + (x * x) + (y * y));
}

__device__ __forceinline__  double x_next(double x, double y, double t)
{
  return 1.0 + u * (x * std::cos(t) - y * std::sin(t));
}

__device__ __forceinline__  double y_next(double x, double y, double t)
{
  return u * (x * std::sin(t) + y * std::cos(t));
}

__device__ void SetIkedaGPU(int x, int y, uchar* data)
{
  size_t index = y * PIXELDIM + x;
  if (index < PIXELDIM2)
  {
    data[index + PIXELDIM2 + PIXELDIM2] = 0xff;
    data[index + PIXELDIM2] = 0x00;
    data[index] = 0x00;
  }
}

__global__ void IkedaGPUCalc(uchar *d_DataOut)
{
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;

  if (!(tx >= 0 && tx < PIXELDIM) || !(ty >= 0 && ty < PIXELDIM))
    return;

  double zx = (double)tx;
  double zy = (double)ty;

  int iteration = 0;
  while (iteration < MAX_ITERATIONS_IKEDA)
  {
    double t = t_next(zx, zy);
    double x_n = x_next(zx, zy, t);
    double y_n = y_next(zx, zy, t);
    zx = x_n;
    zy = y_n;

   if (iteration > 100)
     SetIkedaGPU((int)ceil((PIXELDIM * (zx - min_x)) / (max_x - min_x)), (int)ceil((PIXELDIM * (zy - min_y)) / (max_y - min_y)), d_DataOut);

   ++iteration;
  }
  __syncthreads();
}

#elif defined (FLOAT_VERSION)

__constant__ float min_x = -0.5f, min_y = -2.5f;
__constant__ float max_x = 6.5f, max_y = 6.5f;
__constant__ float u = 0.918f;

__device__ __forceinline__ float t_next(float x, float y)
{
	return 0.4f - 6.0f / (1.0f + (x * x) + (y * y));
}

__device__ __forceinline__  float x_next(float x, float y, float t)
{
	return 1.0f + u * (x * std::cosf(t) - y * std::sinf(t));
}

__device__ __forceinline__  float y_next(float x, float y, float t)
{
	return u * (x * std::sinf(t) + y * std::cosf(t));
}

__device__ void SetIkedaGPU(int x, int y, uchar* data)
{
	size_t index = y * PIXELDIM + x;
	if (index < PIXELDIM2)
	{
		data[index + PIXELDIM2 + PIXELDIM2] = 0xff;
		data[index + PIXELDIM2] = 0x00;
		data[index] = 0x00;
	}
}

__global__ void IkedaGPUCalc(uchar *d_DataOut)
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;

	if (!(tx >= 0 && tx < PIXELDIM) || !(ty >= 0 && ty < PIXELDIM))
		return;

	float zx = (float)tx;
	float zy = (float)ty;

	int iteration = 0;
	while (iteration < MAX_ITERATIONS_IKEDA)
	{
		float t = t_next(zx, zy);
		float x_n = x_next(zx, zy, t);
		float y_n = y_next(zx, zy, t);
		zx = x_n;
		zy = y_n;

		if (iteration > 100)
			SetIkedaGPU((int)ceil((PIXELDIM * (zx - min_x)) / (max_x - min_x)), (int)ceil((PIXELDIM * (zy - min_y)) / (max_y - min_y)), d_DataOut);

		++iteration;
	}
	__syncthreads();
}
#endif

void IkedaGPU(uchar* data)
{
	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 DimGrid((uint)ceil(((float)PIXELDIM) / BLOCK_SIZE), (uint)ceil(((float)PIXELDIM) / BLOCK_SIZE), 1);

	uchar* data_gpu;
	checkCudaErrors(cudaMalloc((void**)&data_gpu, PIXELDIM3 * sizeof(uchar)));
	checkCudaErrors(cudaMemcpy((void*)data_gpu, (void*)data, PIXELDIM3 * sizeof(uchar), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaGetLastError());

	IkedaGPUCalc << <DimGrid, DimBlock >> >(data_gpu);
	checkCudaErrors(cudaGetLastError());

	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy((void*)data, (void*)data_gpu, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(data_gpu));
}
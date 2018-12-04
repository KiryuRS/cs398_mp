#include "Common.h"


#define BrotSize 32
#define iterationBS 64 
#define magBS 2.5
#define shiftBS -PIXELDIM/2.0
#define shiftBS2 -PIXELDIM/1.5

__global__ void MandrelbrotKernel(uchar* d_DataOut, uint limit)
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;

	if (tx >= PIXELDIM || ty >= PIXELDIM)
		return;

	double a = 0.0;
	double b = 0.0;
	double norm2 = 0.0;
	int n;
	double x = (double)((tx + shiftBS2) *magBS) / PIXELDIM;
	double y = (double)((PIXELDIM - 1 - ty + shiftBS) *magBS) / PIXELDIM;
	double iter = 1;

	for (n = 0; norm2 < 4.0 && n < iterationBS; ++n)
	{
		double c = a*a - b*b + x;
		b = 2.0 * a * b + y;
		a = c;
		norm2 = a*a + b*b;

	}
	iter -= double(n) / iterationBS;
	int val = (int)(255 * iter);


	if (val == 0)
	{
		d_DataOut[tx + PIXELDIM * ty] = val; // b
		d_DataOut[tx + PIXELDIM * ty + PIXELDIM2] = val; // g
		d_DataOut[tx + PIXELDIM * ty + PIXELDIM2 + PIXELDIM2] = val; // r
	}
	else
	{
		d_DataOut[tx + PIXELDIM * ty] = val; // b
		d_DataOut[tx + PIXELDIM * ty + PIXELDIM2] = val; // g
		d_DataOut[tx + PIXELDIM * ty + PIXELDIM2 + PIXELDIM2] = val; // r
	}
}

#define MandrelbrotDefault
void Mandrelbrot::MandrelbrotGPU(uchar** gpuOutput)
{
#ifdef MandrelbrotDefault


	dim3 Block(BrotSize, BrotSize, 1);
	dim3 Grid(ceil(((float)PIXELDIM) / BrotSize), ceil(((float)PIXELDIM) / BrotSize), 1);


	checkCudaErrors(cudaMalloc((void **)&ptr1, PIXELDIM3 * sizeof(uchar)));
	MandrelbrotKernel << <Grid, Block >> > (ptr1, PIXELDIM);

	cudaDeviceSynchronize();

	*gpuOutput = (uchar *)malloc(PIXELDIM3 * sizeof(uchar));
	checkCudaErrors(cudaMemcpy(*gpuOutput, ptr1, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));
	//uchar* d_DataIn = nullptr;
	//uchar* d_DataOut = nullptr;
	//
	//cudaMalloc((void **)&d_DataIn, PIXELDIM3);
	//cudaMalloc((void **)&d_DataOut, PIXELDIM3);
	//cudaMemcpy(d_DataIn, cpuOutput, PIXELDIM3, cudaMemcpyHostToDevice);
	//
	////Calls kernel function for mandrelbrot
	//
	//cudaMemcpy(*gpuOutput, d_DataOut, PIXELDIM3, cudaMemcpyDeviceToHost);
#endif
}
void Mandrelbrot::ClearMemory(uchar** data)
{
#ifdef MandrelbrotDefault

	cudaFree(ptr1);
	free(*data);


#endif
}

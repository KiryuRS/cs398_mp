#include "Common.h"


#define BrotSize 32



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
	double x = (double)((tx + shiftM2) *magM) / PIXELDIM;
	double y = (double)((PIXELDIM - 1 - ty + shiftM) *magM) / PIXELDIM;
	double iter = 1;

	for (n = 0; norm2 < 4.0 && n < iterationM; ++n)
	{
		double c = a*a - b*b + x;
		b = 2.0 * a * b + y;
		a = c;
		norm2 = a*a + b*b;

	}
	iter -= double(n) / iterationM;
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
#endif
}
void Mandrelbrot::ClearMemory(uchar** data)
{
#ifdef MandrelbrotDefault

	cudaFree(ptr1);
	free(*data);


#endif
}

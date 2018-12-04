#include "MandrelBrot.h"
#include <iostream>
//#define iterationBS 64 
//#define magBS 0.2
//#define shiftBS -PIXELDIM/1.6
//#define shiftBS2 -PIXELDIM/0.111

#define iterationBS 64 
#define magBS 2.5
#define shiftBS -PIXELDIM/2.0
#define shiftBS2 -PIXELDIM/1.5

void MandrelBrotSetData(int x, int y, int value, uchar* data)
{
	if (value == 0)
	{
		data[x + PIXELDIM * y] = value; // b
		data[x + PIXELDIM * y + PIXELDIM2] = value; // g
		data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = value; // r
	}
	else
	{
		data[x + PIXELDIM * y] = value; // b
		data[x + PIXELDIM * y + PIXELDIM2] = value; // g
		data[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0xBF; // r
	}
}

void MandrelbrotGPU(uchar* cpuOutput, uchar** gpuOutput)
{
	uchar* d_DataIn = nullptr;
	uchar* d_DataOut = nullptr;

	cudaMalloc((void **)&d_DataIn, PIXELDIM3);
	cudaMalloc((void **)&d_DataOut, PIXELDIM3);
	cudaMemcpy(d_DataIn, cpuOutput, PIXELDIM3, cudaMemcpyHostToDevice);

	//Calls kernel function for mandrelbrot

	cudaMemcpy(*gpuOutput, d_DataOut, PIXELDIM3, cudaMemcpyDeviceToHost);
}



void MandrelbrotCPU(uchar* data)
{
	int value = 0;
	for (int j = 0; j < PIXELDIM; ++j) 
	{
		for (int i = 0; i < PIXELDIM; ++i)
		{

			double a = 0.0, b = 0.0, norm2 = 0.0;
			int n;
			double x = static_cast<double>(i + shiftBS2) *magBS / PIXELDIM;
			double y = static_cast<double>(PIXELDIM - 1 - j + shiftBS) *magBS / PIXELDIM;
			for (n = 0; norm2 < 4.0 && n < iterationBS; ++n) 
			{
				double c = a*a - b*b + x;
				b = 2.0*a*b + y;
				a = c;
				norm2 = a*a + b*b;
			
			}
			int value = int(255 * (1 - double(n) / iterationBS));
		
			MandrelBrotSetData(i, j, value, data);
		}
		
	}
}

//__device__ int Mandrelbrot(int x, int y, int maxIter)
//{
//
//}
//
//__global__ void MandrelbrotKernel(uchar* cpu, uchar* gpu)
//{
//
//}
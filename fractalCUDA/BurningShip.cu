#include "Common.h"

#define shipBlock_size 32

__device__
double fastAbsD(double in)
{
	return fabs(in);
}



__global__ void BurningShipDefaultCu(uchar *d_DataOut, uint limit)
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;
	//int value = 0;

	if (tx >= PIXELDIM || ty >= PIXELDIM)
		return;

	double a = 0.0;
	double b = 0.0;
	double norm2 = 0.0;
	int n;
	double x = (double)((tx + shiftBS2) *magBS) / PIXELDIM;
	double y = (double)((PIXELDIM - 1 - ty + shiftBS) *magBS) / PIXELDIM;
	double iter = 1;
	//std::cout << " start " << x << " " << y << std::endl;
	for (n = 0; norm2 < 4.0 && n < iterationBS; ++n) {
		
		 double c = a*a - b*b + x;
			b = 2.0*fabs(a*b) + y;
			a = c;
			norm2 = a*a + b*b;
		
		//std::cout << a << " " << b << std::endl;
		//std::cout << " s " << n << " " << norm2 << std::endl;
	}
	iter -= double(n) / iterationBS;
	int value = (int)(255 * iter);
	
	
	
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



#define BurningshipDefault

//-fmad=false command line to fix precision issue due to default optimize by compiler

void BurningShip::BurningShipGPU(uchar** data)
{
#ifdef BurningshipDefault


	dim3 DimBlock(shipBlock_size, shipBlock_size, 1);
	dim3 DimGrid(ceil(((float)PIXELDIM) / shipBlock_size), ceil(((float)PIXELDIM) / shipBlock_size), 1);


	checkCudaErrors(cudaMalloc((void **)&ptr1, PIXELDIM3 * sizeof(uchar)));
	BurningShipDefaultCu << <DimGrid, DimBlock >> > (ptr1, PIXELDIM);
	//
	//
	cudaDeviceSynchronize();

	*data = (uchar *)malloc(PIXELDIM3 * sizeof(uchar));
	checkCudaErrors(cudaMemcpy(*data, ptr1, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));

#endif




}

void BurningShip::clearGPUMemory(uchar** data)
{
#ifdef BurningshipDefault

	cudaFree(ptr1);
	free(*data);


#endif
}
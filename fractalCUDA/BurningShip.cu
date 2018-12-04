#include "Common.h"

#define shipBlock_size 32

#define BurningshipUnified			// BurningshipDefault, BurningshipUnified, BurningshipPinned
#define BurnInstrinic				// BurnDefault,BurnInstrinic

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
			//b = 2.0*fabs(a*b) + y;
		 b = 2.0*a*b;
		 b = b < 0 ? -b : b;
		 b += y;
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

__device__ double getC(double a, double b, double c)
{
	return __dsub_rd(__fma_rd(a,a,c) ,__dmul_rd(b, b));
}

__device__ double getb(double a, double b, double c)
{
	return __fma_rd(2.0, fabs(__dmul_rd(a, b)), c);
}

__device__ double getNorm(double a, double b)
{
	return __fma_rd(a, a, __dmul_rd(b, b));
}


__device__ double simpleMulti(double a, double b)
{
	return __dmul_rd(a, b);
}


__device__ double getIter(double n, double total)
{
	return __dsub_rd(1.0, __ddiv_rd(n, total));
}

__device__ double valueGet(double n, double total)
{
	return __double2int_rd(__dmul_rd(255.0, __dsub_rd(1.0, __ddiv_rd(n, total))));
}

__device__ double getX(double a, double b, double c, double d)
{
	return __ddiv_rd(__dmul_rd(__dadd_rd(a, b), c), d);
}

__device__ double getY(double a, double b, double c, double d, double e)
{
	return __ddiv_rd(__dmul_rd(__dsub_rd(__dadd_rd(a, c), ++b), d), e);
}


__global__ void BurningShipIntrinsicsCu(uchar *d_DataOut, uint limit)
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
	double x = getX(tx, shiftBS2, magBS, PIXELDIM);
	double y = getY(PIXELDIM, ty, shiftBS, magBS, PIXELDIM);
	for (n = 0; norm2 < 4.0 && n < iterationBS; ++n) {

		double c = getC(a,b,x);
		b = getb(a, b, y);
		a = c;
		norm2 = getNorm(a,b);
	}

	//double iter = getIter(n, iterationBS);
	int value = valueGet(n, iterationBS);


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
#endif
	//
	//
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
#endif
	//
	//
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
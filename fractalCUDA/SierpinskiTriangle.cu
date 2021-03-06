/******************************************************************************/
/*!
@file   SierpinskiTriangle.cu
@par    Purpose: Implementation of SierpinskiTriangle CUDA kernel
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author YongKiat
@par    Email: yongkiat.ong\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/
#include "Common.h"


#define TriangleSize (1<< 5)


__global__ void SierpinskiTriangleKernel(uchar* d_DataOut,uint limit)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y* blockDim.y;

	//if (x >= PIXELDIM || y >= PIXELDIM)
	//	return;
	int i = 0;
	for (; y >= 0; --y)
	{

		// printing space till 
		// the value of y 
		for (; i < y; ++i)
		{
			//outFile << " ";
			d_DataOut[x  + PIXELDIM *  y] = 0XFF; // b
			d_DataOut[x  + PIXELDIM * y + PIXELDIM2] = 0XFF; // g
			d_DataOut[x  + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0XFF; // r
		

		}
		// printing '*' 
		for (; x + y < PIXELDIM;++x)
		{

			// printing '*' at the appropriate position 
			// is done by the and value of x and y 
			// wherever value is 0 we have printed '*' 
			if ((x&y))
			{

				d_DataOut[x + PIXELDIM * y] = 0XFF; // b
				d_DataOut[x + PIXELDIM * y + PIXELDIM2] = 0XFF; // g
				d_DataOut[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0XFF; // r
				//outFile << " " << " ";
				//SetData(x, y, 255, data);
				//SetData(x, y, 255, data);
			}
			else
			{
				d_DataOut[x + PIXELDIM * y] = 0x00; // b
				d_DataOut[x + PIXELDIM * y + PIXELDIM2] = 0x00; // g
				d_DataOut[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0xff; // r
				//SetData(x, y, 0, data);
				//outFile << "* "; 
				
			}

		}

		//outFile << endl; 
	}


}



#define STriangleDefault
void STriangle::TriangleGPU(uchar** gpuOutput)
{
#ifdef STriangleDefault

	dim3 Block(TriangleSize, TriangleSize, 1);
	dim3 Grid(ceil(((float)PIXELDIM) / TriangleSize), ceil(((float)PIXELDIM) / TriangleSize), 1);


	checkCudaErrors(cudaMalloc((void **)&ptr1, PIXELDIM3 * sizeof(uchar)));
	SierpinskiTriangleKernel << <Grid, Block >> > (ptr1,PIXELDIM);

	cudaDeviceSynchronize();

	*gpuOutput = (uchar *)malloc(PIXELDIM3 * sizeof(uchar));
	checkCudaErrors(cudaMemcpy(*gpuOutput, ptr1, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));


#endif
}
void STriangle::ClearMemory(uchar**data)
{
#ifdef STriangleDefault

	cudaFree(ptr1);
	free(*data);


#endif
}
#include "Common.h"


#define TriangleSize (1<< 5)


__global__ void SierpinskiTriangleKernel(uchar* d_DataOut,uint limit)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y* blockDim.y;

	if (x >= PIXELDIM || y >= PIXELDIM)
		return;

	for (; y >= 0; y--)
	{

		// printing space till 
		// the value of y 
		for (int i = 0; i < y; i++)
		{
			//outFile << " ";
			d_DataOut[x + PIXELDIM *  y] = 0xff; // b
			d_DataOut[x  + PIXELDIM * y + PIXELDIM2] = 0xff; // g
			d_DataOut[x  + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0xff; // r
		

		}
		// printing '*' 
		for (; x + y < PIXELDIM; x++)
		{

			// printing '*' at the appropriate position 
			// is done by the and value of x and y 
			// wherever value is 0 we have printed '*' 
			if (x & y)
			{

				d_DataOut[x + PIXELDIM * y] = 0xff; // b
				d_DataOut[x + PIXELDIM * y + PIXELDIM2] = 0xff; // g
				d_DataOut[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0xff; // r
				//outFile << " " << " ";
				//SetData(x, y, 255, data);
				//SetData(x, y, 255, data);
			}
			else
			{
				d_DataOut[x + PIXELDIM * y] = 0x00; // b
				d_DataOut[x + PIXELDIM * y + PIXELDIM2] = 0x00; // g
				d_DataOut[x + PIXELDIM * y + PIXELDIM2 + PIXELDIM2] = 0x00; // r
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
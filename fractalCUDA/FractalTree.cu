#include "Common.h"

#define FractalTreeIterative			// FractalTreeDefault, FractalTreeIterative


__device__ void FractalTreeGPULineDraw(int x1, int y1, int x2, int y2, uchar* data, uint counter)
{
	int dx = x2 - x1;
	int dy = y2 - y1;

	counter *= 5;

	if (std::abs(dy) > std::abs(dx))
	{
		int absY = std::abs(dy);
		float deltaX = static_cast<float>(dx) / absY;

		float startX = x1 - deltaX;

		if (y2 > y1)
		{
			for (int y_i = y1; y_i < y2; y_i++)
			{
				startX += deltaX;

				data[static_cast<int>(startX) + PIXELDIM * y_i] = 0x00; // b
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2] = counter; // g
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2 + PIXELDIM2] = 255 - counter; // r
			}
		}
		else
		{
			for (int y_i = y1; y_i > y2; y_i--)
			{
				startX += deltaX;
				data[static_cast<int>(startX) + PIXELDIM * y_i] = 0x00; // b
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2] = counter; // g
				data[static_cast<int>(startX) + PIXELDIM * y_i + PIXELDIM2 + PIXELDIM2] = 255 - counter; // r
			}
		}

	}
	else
	{
		int absX = std::abs(dx);
		float deltaY = static_cast<float>(dy) / absX;

		float startY = y1 - deltaY;

		if (x2 > x1)
		{
			for (int x_i = x1; x_i < x2; x_i++)
			{
				startY += deltaY;
				data[x_i + PIXELDIM * static_cast<int>(startY)] = 0x00; // b
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2] = counter; // g
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2 + PIXELDIM2] = 255 - counter; // r
			}
		}
		else
		{
			for (int x_i = x1; x_i > x2; x_i--)
			{
				startY += deltaY;
				data[x_i + PIXELDIM * static_cast<int>(startY)] = 0x00; // b
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2] = counter; // g
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2 + PIXELDIM2] = 255 - counter; // r
			}
		}
	}

}

__device__ void RecursionFractalTreeGPU(int locX, int locY, float VecX, float VecY, uchar* data, uint counter)
{
	float length = std::sqrtf(VecX*VecX + VecY*VecY);
	if (std::sqrtf(VecX*VecX + VecY*VecY) < lim)
		return;
	float vecX_n = VecX / length;
	float vecY_n = VecY / length;

	float d1_x = r_xcos(vecX_n, vecY_n);
	float d1_y = r_ycos(vecX_n, vecY_n);




	float d2_x = nr_xcos(vecX_n, vecY_n);
	float d2_y = nr_ycos(vecX_n, vecY_n);

	//std::cout << d1_x << " " << d1_y << " h " << d2_x << " " << d2_y << std::endl;

	length *= per;

	FractalTreeGPULineDraw(locX, locY, locX + d1_x * length, locY + d1_y * length, data, counter + 1);
	FractalTreeGPULineDraw(locX, locY, locX + d2_x * length, locY + d2_y * length, data, counter + 1);

	RecursionFractalTreeGPU(locX + d1_x * length, locY + d1_y * length, d1_x * length, d1_y * length, data, counter + 1);
	RecursionFractalTreeGPU(locX + d2_x * length, locY + d2_y * length, d2_x * length, d2_y * length, data, counter + 1);
	//std::cout << counter + 1 << std::endl;
}


__global__ void FractalTreeDefaultFn(uchar *d_DataOut, uint limit)
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	//int value = 0;

	if (tx < limit)
	{
		float x0 = PIXELDIM / 2.0f;
		float y0 = sY;

		float x1 = PIXELDIM / 2.0f;
		float y1 = eY;

		float dx = x1 - x0;
		float dy = y1 - y0;

		uint counter = 0;

		FractalTreeGPULineDraw(x0, y0, x1, y1, d_DataOut, counter);

		RecursionFractalTreeGPU(x1, y1, dx, dy, d_DataOut, counter);
	}


}


/*

__device__ uint FractalIterativeCounter(float dist)
{
	uint count = 0;

	while (dist >= 3.0f)
	{
		count++;
		dist *= 13.0f / 16.0f;

	}
	return count;
}
*/

struct FracTreeGPU
{
	float locX;
	float locY;
	float vecX;
	float vecY;


};




__global__ void FractalTreeGPUiterative(uchar *d_DataOut, uint limit, float * fpt1, float *fpt2, uint maxLimit, uint itr)
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	//int value = 0;

	uint tpos = tx * 4;

	if (tx < limit)
	{
		float newPosx = fpt1[tpos] + fpt1[tpos + 2];
		float newPosy = fpt1[tpos+1] + fpt1[tpos + 3];
		FractalTreeGPULineDraw(fpt1[tpos], fpt1[tpos+1], newPosx, newPosy, d_DataOut, itr);

		if (itr != maxLimit)
		{
			float vecX = fpt1[tpos + 2];
			float vecY = fpt1[tpos + 3];

			float length = std::sqrtf(vecX* vecX + vecY * vecY);


			float vecX_n = vecX / length;
			float vecY_n = vecY / length;

			length *= per;

			float d1_x = (r_xcos(vecX_n, vecY_n)) * length;
			float d1_y = (r_ycos(vecX_n, vecY_n)) * length;

			float d2_x = (nr_xcos(vecX_n, vecY_n)) * length;
			float d2_y = (nr_ycos(vecX_n, vecY_n)) * length;

			int startPoint = tpos * 2;

			fpt2[startPoint] = newPosx;
			fpt2[startPoint + 1] = newPosy;
			fpt2[startPoint + 2] = d1_x;
			fpt2[startPoint + 3] = d1_y;
			fpt2[startPoint + 4] = newPosx;
			fpt2[startPoint + 5] = newPosy;
			fpt2[startPoint + 6] = d2_x;
			fpt2[startPoint + 7] = d2_y;

		}

	}



}

__global__ void setUpFirstVariable(float * fptr1)
{
	fptr1[0] = PIXELDIM / 2.0f;
	fptr1[1] = sY;
	fptr1[2] = 0.0f;
	fptr1[3] = eY -sY;

}


void FractalTree::FractalTreeGPU(uchar** data)
{

	//dim3 DimBlock(shipBlock_size, shipBlock_size, 1);
	//dim3 DimGrid(ceil(((float)PIXELDIM) / shipBlock_size), ceil(((float)PIXELDIM) / shipBlock_size), 1);

#ifdef FractalTreeDefault


	checkCudaErrors(cudaMalloc((void **)&ptr1 , PIXELDIM3 * sizeof(uchar)));
	cudaMemset(ptr1, 0xFF, PIXELDIM3 * sizeof(uchar));
	size_t limit = 2000;

	cudaDeviceSetLimit(cudaLimitStackSize, limit);
	FractalTreeDefaultFn <<<1,1>>>(ptr1, 1);


	*data = (uchar *)malloc(PIXELDIM3 * sizeof(uchar));
	checkCudaErrors(cudaMemcpy(*data, ptr1, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));


	cudaDeviceSynchronize();
#elif defined FractalTreeIterative




	uint count = 0;
	float dist = eY - sY;
	while (dist >= lim)
	{
		count++;
		dist *= per;

	}

	uint maxNode = 0x01 << count;
	count++;
	checkCudaErrors(cudaMalloc((void **)&ptr1, PIXELDIM3 * sizeof(uchar)));
	cudaMemset(ptr1, 0xFF, PIXELDIM3 * sizeof(uchar));
	checkCudaErrors(cudaMalloc((void **)&fptr1, maxNode * sizeof(FracTreeGPU)));
	checkCudaErrors(cudaMalloc((void **)&fptr2, maxNode * sizeof(FracTreeGPU)));
	//cudaMemset(ptr1, 0xFF, PIXELDIM3 * sizeof(uchar));

	//FractalTreeGPUiterative << <1, 1 >> >(ptr1, 1);

	setUpFirstVariable << <1, 1 >> > (fptr1);


	//fptr1[0] = PIXELDIM / 2.0f;
	//fptr1[1] = 50.0f;
	//fptr1[2] = 0.0f;
	//fptr1[3] = 70.0f;
	
	for (uint itr = 0; itr < count; itr++)
	{
		uint i = 0x01 << itr;

		FractalTreeGPUiterative << <ceil(((float)i) / fTsingleBlock), fTsingleBlock >> >(ptr1, i, fptr1, fptr2, count - 1, itr);
	
		//cudaDeviceSynchronize();
		//checkCudaErrors(cudaMemcpy(fptr1, fptr2, i * sizeof(FracTreeGPU), cudaMemcpyDeviceToDevice));

		float * tmp = fptr1;
		fptr1 = fptr2;
		fptr2 = tmp;
		//std::cout << i << std::endl;
	
	}






	*data = (uchar *)malloc(PIXELDIM3 * sizeof(uchar));
	checkCudaErrors(cudaMemcpy(*data, ptr1, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	//uint counter = FractalIterativeCounter << <1, 1 >> > (120 - 50);

	//std::cout << counter << std::endl;

#endif




}


void FractalTree::clearGPUMemory(uchar** data)
{
#ifdef FractalTreeDefault
	free(*data);
	checkCudaErrors(cudaFree(ptr1));
#elif defined FractalTreeIterative
	free(*data);
	checkCudaErrors(cudaFree(ptr1));
	checkCudaErrors(cudaFree(fptr1));
	checkCudaErrors(cudaFree(fptr2));

#endif
}
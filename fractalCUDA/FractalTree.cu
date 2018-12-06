#include "Common.h"

#define FractalTreeIterative2			// FractalTreeDefault, FractalTreeIterative, FractalTreeIterativeSM, FractalTreeIterative2


__device__ void FractalTreeGPULineDraw(int x1, int y1, int x2, int y2, uchar* data, uint counter)
{
#ifdef fTdraw
	/// simple iteration of line  for drawing
	int dx = x2 - x1;
	int dy = y2 - y1;

	counter *= 5;
	//counter = counter > 255 ? 255 : 0;

	/// check the abs different
	if (std::abs(dy) > std::abs(dx))
	{
		/// get the dx base on y shift
		int absY = std::abs(dy);
		float deltaX = static_cast<float>(dx) / absY;

		float startX = x1 - deltaX;

		if (y2 > y1)
		{
			/// increment and draw
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
			/// increment and draw
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
		/// get the dy base on x shift
		int absX = std::abs(dx);
		float deltaY = static_cast<float>(dy) / absX;

		float startY = y1 - deltaY;

		if (x2 > x1)
		{
			/// increment and draw
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
			/// increment and draw
			for (int x_i = x1; x_i > x2; x_i--)
			{
				startY += deltaY;
				data[x_i + PIXELDIM * static_cast<int>(startY)] = 0x00; // b
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2] = counter; // g
				data[x_i + PIXELDIM * static_cast<int>(startY) + PIXELDIM2 + PIXELDIM2] = 255 - counter; // r
			}
		}
	}
#endif
}

__device__ void RecursionFractalTreeGPU(int locX, int locY, float VecX, float VecY, uchar* data, uint counter)
{
	/// check for the length
	float length = std::sqrtf(VecX*VecX + VecY*VecY);
	if (std::sqrtf(VecX*VecX + VecY*VecY) < lim)
		return;
	/// normalized
	float vecX_n = VecX / length;
	float vecY_n = VecY / length;

	/// get the new vector
	float d1_x = r_xcos(vecX_n, vecY_n);
	float d1_y = r_ycos(vecX_n, vecY_n);


	float d2_x = nr_xcos(vecX_n, vecY_n);
	float d2_y = nr_ycos(vecX_n, vecY_n);

	/// set the new length
	length *= per;


	/// draw the line
	FractalTreeGPULineDraw(locX, locY, locX + d1_x * length, locY + d1_y * length, data, counter + 1);
	FractalTreeGPULineDraw(locX, locY, locX + d2_x * length, locY + d2_y * length, data, counter + 1);

	/// recur to next two line
	RecursionFractalTreeGPU(locX + d1_x * length, locY + d1_y * length, d1_x * length, d1_y * length, data, counter + 1);
	RecursionFractalTreeGPU(locX + d2_x * length, locY + d2_y * length, d2_x * length, d2_y * length, data, counter + 1);
}


__global__ void FractalTreeDefaultFn(uchar *d_DataOut, uint limit)
{
	/// set up for first line
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	//int value = 0;

	if (tx < limit)
	{
		/// set up line data
		float x0 = PIXELDIM / 2.0f;
		float y0 = sY;

		float x1 = PIXELDIM / 2.0f;
		float y1 = eY;

		float dx = x1 - x0;
		float dy = y1 - y0;

		/// to use for color
		uint counter = 0;

		/// draw first line
		FractalTreeGPULineDraw(x0, y0, x1, y1, d_DataOut, counter);

		/// call to draw next two line
		RecursionFractalTreeGPU(x1, y1, dx, dy, d_DataOut, counter);
	}


}


/// data structure for the iterative
struct FracTreeGPU
{
	float locX;
	float locY;
	float vecX;
	float vecY;


};



/// iterative for gpu
__global__ void FractalTreeGPUiterative(uchar *d_DataOut, uint limit, float * fpt1, float *fpt2, uint maxLimit, uint itr)
{
	/// get the position
	int tx = threadIdx.x + blockIdx.x * blockDim.x;

	/// get the position - shift based on structure for iteration
	uint tpos = tx * 4;

	/// check if its within current limitation
	if (tx < limit)
	{
		/// store the end point
		float newPosx = fpt1[tpos] + fpt1[tpos + 2];
		float newPosy = fpt1[tpos+1] + fpt1[tpos + 3];

		/// draw the line
		FractalTreeGPULineDraw(fpt1[tpos], fpt1[tpos+1], newPosx, newPosy, d_DataOut, itr);

		/// check if the limit is been reach
		if (itr != maxLimit)
		{
			/// calculate the new vectors
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

			/// set up for next point
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

__global__ void FractalTreeGPUiterativeSM(uchar *d_DataOut, uint limit, float * fpt1, uint maxLimit, uint itr)
{
	/// calculate the position
	int tx = threadIdx.x + blockIdx.x * blockDim.x;

	/// set up for shared block - design to suit the current structure
	__shared__ float shared_thread[fTsharedMem];

	/// calculate shared blk position
	uint tshare = threadIdx.x * 4;
	uint tpos = tx * 4;

	/// check the current iteration limit
	if (tx < limit)
	{
		/// data storage
		shared_thread[tshare]	= fpt1[tpos];
		shared_thread[tshare+1] = fpt1[tpos+1];
		shared_thread[tshare+2] = fpt1[tpos+2];
		shared_thread[tshare+3] = fpt1[tpos+3];
		
		/// get the new position
		float newPosx = shared_thread[tshare] + shared_thread[tshare + 2];
		float newPosy = shared_thread[tshare + 1] + shared_thread[tshare + 3];

		/// draw
		FractalTreeGPULineDraw(shared_thread[tshare], shared_thread[tshare + 1], newPosx, newPosy, d_DataOut, itr);
		if (itr != maxLimit)
		{
			/// sync threads
			__syncthreads();
			
			/// calculate the new vector
			float vecX = shared_thread[tshare + 2];
			float vecY = shared_thread[tshare + 3];

			float length = std::sqrtf(vecX* vecX + vecY * vecY);


			float vecX_n = vecX / length;
			float vecY_n = vecY / length;

			length *= per;

			float d1_x = (r_xcos(vecX_n, vecY_n)) * length;
			float d1_y = (r_ycos(vecX_n, vecY_n)) * length;

			float d2_x = (nr_xcos(vecX_n, vecY_n)) * length;
			float d2_y = (nr_ycos(vecX_n, vecY_n)) * length;

			/// different storage point compare to version 1 of iterative
			int startPoint = limit * 4 + tpos;

			fpt1[tpos] = newPosx;
			fpt1[tpos + 1] = newPosy;
			fpt1[tpos + 2] = d1_x;
			fpt1[tpos + 3] = d1_y;
			fpt1[startPoint] = newPosx;
			fpt1[startPoint + 1] = newPosy;
			fpt1[startPoint + 2] = d2_x;
			fpt1[startPoint + 3] = d2_y;

		}

	}



}

__global__ void FractalTreeGPUiterative2(uchar *d_DataOut, uint limit, float * fpt1, uint maxLimit, uint itr)
{
	/// calculate the position
	int tx = threadIdx.x + blockIdx.x * blockDim.x;


	uint tpos = tx * 4;

	/// check the current iteration limit
	if (tx < limit)
	{
		/// store the end point
		float newPosx = fpt1[tpos] + fpt1[tpos + 2];
		float newPosy = fpt1[tpos + 1] + fpt1[tpos + 3];

		/// draw the line
		FractalTreeGPULineDraw(fpt1[tpos], fpt1[tpos + 1], newPosx, newPosy, d_DataOut, itr);
		if (itr != maxLimit)
		{
			/// calculate the new vectors
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

			/// different storage point compare to version 1 of iterative
			int startPoint = limit * 4 + tpos;

			fpt1[tpos] = newPosx;
			fpt1[tpos + 1] = newPosy;
			fpt1[tpos + 2] = d1_x;
			fpt1[tpos + 3] = d1_y;
			fpt1[startPoint] = newPosx;
			fpt1[startPoint + 1] = newPosy;
			fpt1[startPoint + 2] = d2_x;
			fpt1[startPoint + 3] = d2_y;

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

	setUpFirstVariable << <1, 1 >> > (fptr1);
	
	for (uint itr = 0; itr < count; itr++)
	{
		uint i = 0x01 << itr;

		FractalTreeGPUiterative << <ceil(((float)i) / fTsingleBlock), fTsingleBlock >> >(ptr1, i, fptr1, fptr2, count - 1, itr);

		float * tmp = fptr1;
		fptr1 = fptr2;
		fptr2 = tmp;
	}

	*data = (uchar *)malloc(PIXELDIM3 * sizeof(uchar));
	checkCudaErrors(cudaMemcpy(*data, ptr1, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();


#elif defined FractalTreeIterativeSM



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

	setUpFirstVariable << <1, 1 >> > (fptr1);

	for (uint itr = 0; itr < count; itr++)
	{
		uint i = 0x01 << itr;

		FractalTreeGPUiterativeSM << <ceil(((float)i) / fTsingleBlock), fTsingleBlock >> >(ptr1, i, fptr1, count - 1, itr);

	}

	*data = (uchar *)malloc(PIXELDIM3 * sizeof(uchar));
	checkCudaErrors(cudaMemcpy(*data, ptr1, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();


#elif defined FractalTreeIterative2


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

	setUpFirstVariable << <1, 1 >> > (fptr1);

	for (uint itr = 0; itr < count; itr++)
	{
		uint i = 0x01 << itr;

		FractalTreeGPUiterative2 << <ceil(((float)i) / fTsingleBlock), fTsingleBlock >> >(ptr1, i, fptr1, count - 1, itr);

	}

	*data = (uchar *)malloc(PIXELDIM3 * sizeof(uchar));
	checkCudaErrors(cudaMemcpy(*data, ptr1, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

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

#elif defined FractalTreeIterativeSM
	free(*data);
	checkCudaErrors(cudaFree(ptr1));
	checkCudaErrors(cudaFree(fptr1));
#elif defined FractalTreeIterative2
	free(*data);
	checkCudaErrors(cudaFree(ptr1));
	checkCudaErrors(cudaFree(fptr1));
#endif
}
#include "SierpinskiCarpet.h"

void SierpinskiCarpetCPU(uchar *)
{
	FileWriter fw{ "cpuOutput_KENNETH_Carpet.txt" };

	// Initialization
	int dim = 1, d = 0;
	int depth = SIERPINSKI_DEPTH;
	for (uint i = 0; i != depth; ++i)
		dim *= SIERPINSKI_DEPTH;

	uint width = dim + 2;			// accounting for 2 extra spaces
	uint height = dim + 2;			// just to make it a square
	uint size = width * height;

	// Create the data for printing
	uchar *data = new uchar[size]{ };

	// Performing calculation
	for (int i = 0; i != dim; ++i) {
		for (int j = 0; j != dim + 1; ++j) {
			for (d = dim / SIERPINSKI_DEPTH; d; d /= SIERPINSKI_DEPTH)
				if ((i % (d * SIERPINSKI_DEPTH)) / d == 1 && (j % (d * SIERPINSKI_DEPTH)) / d == 1)
					break;
			// fprintf(file, d ? " " : "#");
			data[i * width + j] = d ? WHITESPACE_PRINT : HEX_PRINT;
		}
		// fprintf(file, "\n");
		data[i * width + dim] = NEWLINE_PRINT;
	}

	// Copy to file
	for (uint i = 0; i != size; ++i)
	{
		switch (data[i])
		{
		case WHITESPACE_PRINT:
			fw.Write("  ");
			break;

		case HEX_PRINT:
			fw.Write("##");
			break;

		case NEWLINE_PRINT:
			fw.Write("\n");
			break;
		}
	}

	// Destruction
	delete[] data;
}

void SierpinskiCarpetGPU(uchar *, uchar **)
{
	FileWriter fw{ "gpuOutput_KENNETH_Carpet.txt" };

	// Initialization
	uchar *d_DataIn = nullptr;
	int dim = 1;
	int depth = SIERPINSKI_DEPTH;
	for (uint i = 0; i != depth; ++i)
		dim *= SIERPINSKI_DEPTH;
	uint width = dim + 2;			// accounting for 2 extra spaces
	uint height = dim + 2;			// just to make it a square
	uint size = width * height;

	// CUDA Creation
#if defined SC_MANAGED
	uchar *data = new uchar[size]{ 0 };
	checkCudaErrors(cudaMalloc((void**)&d_DataIn, size * sizeof(uchar)));

	// Calculation
	SierpinskiCarpetKernel(d_DataIn, width, height);
	cudaDeviceSynchronize();

	// Copying of data
	checkCudaErrors(cudaMemcpy(data, d_DataIn, size * sizeof(uchar), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	// Copy to file
	for (uint i = 0; i != size; ++i)
	{
		switch (data[i])
		{
		case WHITESPACE_PRINT:
			fw.Write("  ");
			break;

		case HEX_PRINT:
			fw.Write("##");
			break;

		case NEWLINE_PRINT:
			fw.Write("\n");
			break;
		}
	}

	// Destruction here
	delete[] data;
	cudaFree(d_DataIn);

#elif defined SC_UNIFIED
	checkCudaErrors(cudaMallocManaged((void**)&d_DataIn, size * sizeof(uchar)));

	// Calculation
	SierpinskiCarpetKernel(d_DataIn, width, height);
	cudaDeviceSynchronize();

	// Copy to file
	for (uint i = 0; i != size; ++i)
	{
		switch (d_DataIn[i])
		{
		case WHITESPACE_PRINT:
			fw.Write("  ");
			break;

		case HEX_PRINT:
			fw.Write("##");
			break;

		case NEWLINE_PRINT:
			fw.Write("\n");
			break;
		}
	}

	// Destruction here
	cudaFree(d_DataIn);

#elif defined SC_PINNED
	checkCudaErrors(cudaMalloc((void**)&d_DataIn, size * sizeof(uchar)));
	uchar *data = nullptr;
	checkCudaErrors(cudaHostAlloc((void**)&data, size * sizeof(uchar), cudaHostAllocDefault));

	// Calculation
	SierpinskiCarpetKernel(d_DataIn, width, height);
	cudaDeviceSynchronize();

	// Copy to memory
	checkCudaErrors(cudaMemcpy(data, d_DataIn, size * sizeof(uchar), cudaMemcpyDeviceToHost));

	// Copy to file
	for (uint i = 0; i != size; ++i)
	{
		switch (data[i])
		{
		case WHITESPACE_PRINT:
			fw.Write("  ");
			break;

		case HEX_PRINT:
			fw.Write("##");
			break;

		case NEWLINE_PRINT:
			fw.Write("\n");
			break;
		}
	}

	cudaFreeHost(data);
	cudaFree(d_DataIn);

#endif
}
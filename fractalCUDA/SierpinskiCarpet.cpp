#include "SierpinskiCarpet.h"

void SierpinskiCarpetCPU(uchar *)
{
	FILE *file = fopen("cpu_sierpinski.txt", "w");
	if (!file)
	{
		std::cerr << "Something went wrong!" << std::endl;
		return;
	}

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
			fprintf(file, " ");
			break;

		case HEX_PRINT:
			fprintf(file, "#");
			break;

		case NEWLINE_PRINT:
			fprintf(file, "\n");
			break;
		}
	}

	// Destruction
	fclose(file);
	delete[] data;
}

void SierpinskiCarpetGPU(uchar * cpuData, uchar ** gpuData)
{
	FILE* file = fopen("gpu_sierpinski.txt", "w");
	if (!file)
	{
		std::cerr << "Something went wrong!" << std::endl;
		return;
	}

	// Initialization
	uchar *d_DataIn = nullptr;
	int dim = 1;
	int depth = SIERPINSKI_DEPTH;
	for (uint i = 0; i != depth; ++i)
		dim *= SIERPINSKI_DEPTH;
	uint width = dim + 2;			// accounting for 2 extra spaces
	uint height = dim + 2;			// just to make it a square
	uint size = width * height;
	uchar *data = new uchar[size]{ 0 };

	// CUDA Creation
	checkCudaErrors(cudaMalloc((void**)&d_DataIn, size * sizeof(uchar)));

	// Calculation
	SierpinskiCarpetKernel(d_DataIn, width, height);
	cudaDeviceSynchronize();

	// Copying of data
	checkCudaErrors(cudaMemcpy(data, d_DataIn, size * sizeof(uchar), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	// Copy to File
	for (uint i = 0; i != size; ++i)
	{
		switch (data[i])
		{
		case WHITESPACE_PRINT:
			fprintf(file, " ");
			break;

		case HEX_PRINT:
			fprintf(file, "#");
			break;

		case NEWLINE_PRINT:
			fprintf(file, "\n");
			break;
		}
	}

	// Destruction here
	fclose(file);
	delete[] data;
	cudaDeviceReset();
}

void SierpinskiCarpetClearGPU(uchar **)
{
	// Do not have to clear since we output to a file
}

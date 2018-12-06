#include "SierpinskiCarpet.h"

void SierpinskiCarpetCPU(uchar *)
{
	FILE *file = fopen("cpu_sierpinski.txt", "w");
	if (!file)
	{
		std::cerr << "Something went wrong!" << std::endl;
		return;
	}

	int dim = 1, d = 0;
	int depth = SIERPINSKI_DEPTH;

	for (uint i = 0; i != depth; ++i)
		dim *= SIERPINSKI_DEPTH;

	// Create the data for printing
	uint size = (dim + 1) * (dim + 2);
	uchar *data = new uchar[size]{};

	for (int i = 0; i != dim; ++i) {
		int j = 0;
		for (; j != dim + 1; ++j) {
			for (d = dim / SIERPINSKI_DEPTH; d; d /= SIERPINSKI_DEPTH)
				if ((i % (d * SIERPINSKI_DEPTH)) / d == 1 && (j % (d * SIERPINSKI_DEPTH)) / d == 1)
					break;
			// fprintf(file, d ? " " : "#");
			data[i * (dim + 2) + j] = d ? WHITESPACE_PRINT : HEX_PRINT;
		}
		// fprintf(file, "\n");
		data[i * (dim + 2) + j] = NEWLINE_PRINT;
	}

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

	fclose(file);
	
	delete[] data;
}

void SierpinskiCarpetGPU(uchar * cpuData, uchar ** gpuData)
{
}

void SierpinskiCarpetClearGPU(uchar **)
{
	// Do not have to clear since we output to a file
}

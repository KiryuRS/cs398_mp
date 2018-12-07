#include "BrownianTree.h"

void BrownianSetData(uint x, uint y, uchar value, uchar* data)
{
	data[y + PIXELDIM * x] = value;								// b
	data[y + PIXELDIM * x + PIXELDIM2] = value;					// g
	data[y + PIXELDIM * x + PIXELDIM2 + PIXELDIM2] = value;		// r
}

void BrownianCPU(uchar* data)
{
	uchar *input = new uchar[PIXELDIM2]{ };

	srand((unsigned)time(nullptr));
	int px, py; // particle values
	int dx, dy; // offsets

	// set the seed
	input[(rand() % PIXELDIM) * PIXELDIM + rand() % PIXELDIM] = 1;

	for (int i = 0; i != BROWNIAN_ITERATIONS; ++i) {
		// set particle's initial position
		px = rand() % PIXELDIM;
		py = rand() % PIXELDIM;

		while (1) {
			// randomly choose a direction
			dx = rand() % 3 - 1;
			dy = rand() % 3 - 1;

			if (dx + px < 0 || dx + px >= PIXELDIM || dy + py < 0 || dy + py >= PIXELDIM) {
				// plop the particle into some other random location
				px = rand() % PIXELDIM;
				py = rand() % PIXELDIM;
			}
			else if (input[(py + dy) * PIXELDIM + px + dx] != 0) {
				// bumped into something
				input[py * PIXELDIM + px] = 1;
				break;
			}
			else {
				py += dy;
				px += dx;
			}
		}
	}

	for (uint i = 0; i != PIXELDIM; ++i)
		for (uint j = 0; j != PIXELDIM; ++j)
		{
			uchar rgb = input[i * PIXELDIM + j] ? 255 : 0;
			BrownianSetData(j, i, rgb, data);
		}

	delete[] input;
}

void BrownianGPU(uchar* cpuData, uchar** gpuData)
{
	return;
	uchar *dataIn = nullptr;
	uchar *dataOut = nullptr;

	// Create memory
	checkCudaErrors(cudaMalloc((void**)&dataIn, PIXELDIM2));
	checkCudaErrors(cudaMallocManaged((void**)&dataOut, PIXELDIM3));
	*gpuData = new uchar[PIXELDIM3];

	// Copy memory to
	checkCudaErrors(cudaMemcpy(dataOut, cpuData, PIXELDIM3, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Randomize the seed
	srand((unsigned)time(nullptr));

	// Call CUDA
	BrownianGPUKernel(dataIn, dataOut);
	cudaDeviceSynchronize();

	// Copy memory over
	checkCudaErrors(cudaMemcpy(*gpuData, dataOut, PIXELDIM3, cudaMemcpyDeviceToHost));

	// Delete memory
	cudaFree(dataIn);
	cudaFree(dataOut);

}

void BrownianClearGPU(uchar ** gpuData)
{
	if (gpuData)
		delete[] *gpuData;
	cudaDeviceReset();
}

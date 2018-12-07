
#include "Common.h"
#include <curand.h>
#include <curand_kernel.h>

struct Lock
{
	int *mutex;

	Lock()
		: mutex{ nullptr }
	{
		int state = 0;
		cudaMalloc((void**)&mutex, sizeof(int));
		cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
	}

	~Lock()
	{
		cudaFree(mutex);
	}

	__device__ void lock()
	{
		while (atomicCAS(mutex, 0, 1) != 0);
	}

	__device__ void unlock()
	{
		atomicExch(mutex, 0);
	}
};

__device__ int mutex = 0;

__global__ void resetMutex(void)
{
	mutex = 0;
}

__global__ void setup_kernel(uchar *data, curandState *state, uint seed)
{
	curand_init(seed, 1, 0, state);
	float _rand = curand_uniform(state) * PIXELDIM;
	uint randy = (uint)truncf(_rand);
	_rand = curand_uniform(state) * PIXELDIM;
	uint randx = (uint)truncf(_rand);

	// Randomly plant a seed
	data[randy * PIXELDIM + randx] = 1;
}

template <typename T>
__device__ void randomDirection(curandState *state, T* value)
{
	float _rand = curand_uniform(state) * PIXELDIM;
	*value = (T)truncf(_rand);
	*value = *value % 3 - 1;
}

template <typename T>
__device__ void randomPosition(curandState *state, T* py, T* px)
{
	float _rand = curand_uniform(state) * PIXELDIM;
	*py = (int)truncf(_rand);
	_rand = curand_uniform(state) * PIXELDIM;
	*px = (int)truncf(_rand);
}

__global__ void heatDistrCalculation(uchar *data, curandState *state, int py, int px)
{
	int dx, dy;

	// Every thread to perform the same operation and determine who add it first
	while (!mutex)
	{
		randomDirection(state, &dx);
		randomDirection(state, &dy);

		int dpx = dx + px;
		int dpy = dy + py;

		if (dpx < 0 || dpx >= PIXELDIM || dpy < 0 || dpy >= PIXELDIM)
			break;

		else if (data[dpy * PIXELDIM + dpx] != 0)
		{
			// Bumped into something
			atomicOr(&mutex, 1);
			data[py * PIXELDIM + px] = 1;
		}
		else
		{
			py += dy;
			px += dx;
		}
	}
	
}

__global__ void heatDistrUpdate(uchar *in, uchar *out)
{
	__shared__ uchar shm[BLOCK_SIZE][BLOCK_SIZE];

	uint tx = threadIdx.x;
	uint ty = threadIdx.y;
	uint x = blockDim.x * blockIdx.x + tx;
	uint y = blockDim.y * blockIdx.y + ty;

	if (x >= PIXELDIM || y >= PIXELDIM)
		return;

	// Copy into our shared memory
	shm[ty][tx] = in[y * PIXELDIM + x] ? 255 : 0;
	__syncthreads();

	// All three colors
	uint rgb = shm[ty][tx];
	out[y * PIXELDIM + x] = rgb;							// B
	out[y * PIXELDIM + x + PIXELDIM2] = rgb;				// G
	out[y * PIXELDIM + x + PIXELDIM2 + PIXELDIM2] = rgb;	// R
}

void BrownianGPUKernel(uchar *d_DataIn, uchar *d_DataOut)
{
	// Setup the variables
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(ceil((float)PIXELDIM / BLOCK_SIZE), ceil((float)PIXELDIM / BLOCK_SIZE));
	
	// Initialize
	srand((uint)time(nullptr));
	curandState *state = nullptr;
	checkCudaErrors(cudaMalloc(&state, sizeof(curandState)));
	setup_kernel<<<1,1>>>(d_DataIn, state, (uint)time(nullptr));

	int py, px;
	// Iterations and calculation
	for (uint i = 0; i != BROWNIAN_ITERATIONS; ++i)
	{
		resetMutex<<<1,1>>>();
		cudaDeviceSynchronize();

		// Set particle's initial position
		py = rand() % PIXELDIM;
		px = rand() % PIXELDIM;

		// Finding the position
		heatDistrCalculation<<<dimGrid, dimBlock>>>(d_DataIn, state, py, px);
		cudaDeviceSynchronize();
	}
	cudaFree(state);

	// Update
	heatDistrUpdate<<<dimGrid, dimBlock>>>(d_DataIn, d_DataOut);
}
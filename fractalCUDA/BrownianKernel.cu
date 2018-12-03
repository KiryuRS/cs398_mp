
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

typedef unsigned char uchar;
typedef unsigned int uint;
#define UINT_BITS					32
#define BLOCK_SIZE					32
#define PIXELDIM		            512
static constexpr size_t PIXELDIM2 = PIXELDIM * PIXELDIM;
static constexpr size_t PIXELDIM3 = PIXELDIM * PIXELDIM * 3;

__global__ void heatDistrCalculation()
{
	
}

__global__ void heatDistrUpdate(uchar *in, uchar *out, uint width, uint height)
{
	uint tx = threadIdx.x;
	uint ty = threadIdx.y;
	uint x = blockDim.x * blockIdx.x + tx;
	uint y = blockDim.y * blockIdx.y + ty;

	// Check if its out of bounds
	if (x >= width || y >= height)
		return;

	// Copy into the data
	uchar rgb = in[y * width + x] ? 255 : 0;

	// All three colors
	out[y * width + x] = rgb;				// B
	out[y * width + x + PIXELDIM2] = rgb;	// G
	out[y * width + x + PIXELDIM3] = rgb;	// R
}

void BrownianGPUKernel(uchar *d_DataIn, uchar *d_DataOut, uint width, uint height)
{
	// Setup the variables
	dim3 UPBLOCK2{ BLOCK_SIZE, BLOCK_SIZE };
	dim3 UPGRID2{ ((float)width + 1) / BLOCK_SIZE - 1, ((float)height + 1) / BLOCK_SIZE - 1 };
	
	// Iterations and calculation

	// Update
	heatDistrUpdate<<<UPGRID2, UPBLOCK2>>>(d_DataIn, d_DataOut, width, height);
}
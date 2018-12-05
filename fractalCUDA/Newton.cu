#include "Common.h"
#include <thrust/complex.h> // mimics std::complex (more natural looking code)
#include <cuComplex.h>

//__device__ cuFloatComplex roots[3] =
//{
//  cuFloatComplex{ 1.0f, 0.0f },
//  cuFloatComplex{ -0.5f,  sqrtf(3.0f) / 2.0f },
//  cuFloatComplex{ -0.5f, -sqrtf(3.0f) / 2.0f }
//};

__device__ thrust::complex<float> Fz(const thrust::complex<float>& z)
{
  return z * z*z - thrust::complex<float>(1.0f, 0.0f);
}

__device__ thrust::complex<float> dFz(const thrust::complex<float>& z)
{
  return thrust::complex<float>(3.0f, 0.0f) * (z*z);
}

__device__ cuFloatComplex cuFz(const cuFloatComplex& z)
{
  return cuCsubf(cuCmulf(cuCmulf(z, z), z), cuFloatComplex{ 1.0f, 0.0f });
}

__device__ cuFloatComplex cudFz(const cuFloatComplex& z)
{
  return cuCmulf(cuFloatComplex{ 3.0f, 0.0f }, cuCmulf(z, z));
}

__device__ void SetDataGPU(int x, int y, uchar* data, int color)
{
  size_t index = y * PIXELDIM + x;
  if (index < PIXELDIM2)
  {
    switch (color)
    {
    case 0:
      data[index + PIXELDIM2 + PIXELDIM2] = 0xff; // r
      data[index + PIXELDIM2] = 0x00;
      data[index] = 0x00;
      break;
    case 1:
      data[index + PIXELDIM2 + PIXELDIM2] = 0x00;
      data[index + PIXELDIM2] = 0xff; // g
      data[index] = 0x00;
      break;
    case 2:
      data[index + PIXELDIM2 + PIXELDIM2] = 0x00;
      data[index + PIXELDIM2] = 0x00;
      data[index] = 0xff; // b
      break;
    }
  }
}

__global__ void NewtonGPUCalc(uchar *d_DataOut)
{
  // __shared__ thrust::complex<float> roots[3];

  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;

  if (tx >= PIXELDIM || ty >= PIXELDIM)
    return;

  float zx = (float)tx * 2.0f / (PIXELDIM - 1) + -1.0f;
  float zy = (float)ty * 2.0f / (PIXELDIM - 1) + -1.0f;

  //// Mapped coordinates
  //thrust::complex<float> z{ zx, zy };

  ////// Roots of polynomials
  //thrust::complex<float> roots[3] =
  //{
  //  thrust::complex<float>{ 1.0f, 0.0f },
  //  thrust::complex<float>{ -0.5f,  sqrtf(3.0f) / 2.0f },
  //  thrust::complex<float>{ -0.5f, -sqrtf(3.0f) / 2.0f }
  //};

    // Mapped coordinates
  cuFloatComplex z{ zx, zy };

  //// Roots of polynomials
  cuFloatComplex roots[3] =
  {
    cuFloatComplex{ 1.0f, 0.0f },
    cuFloatComplex{ -0.5f,  sqrtf(3.0f) / 2.0f },
    cuFloatComplex{ -0.5f, -sqrtf(3.0f) / 2.0f }
  };

  int iteration = 0;
  bool done = false;
  while (iteration < MAX_ITERATIONS && !done)
  {
    z = cuCsubf(z, cuCdivf(cuFz(z), cudFz(z)));

    for (int i = 0; i < 3; ++i)
    {
      // thrust::complex<float> diff = (z - roots[i]);
      cuFloatComplex diff = cuCsubf(z, roots[i]);

      if (std::fabsf(diff.x) < EPSILON && std::fabsf(diff.y) < EPSILON)
      {
        SetDataGPU(tx, ty, d_DataOut, i);
        done = true;
        break;
      }
    }
    ++iteration;
    __syncthreads();
  }
}

void NewtonGPU(uchar* data)
{
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 DimGrid((uint)ceil(((float)PIXELDIM) / BLOCK_SIZE), (uint)ceil(((float)PIXELDIM) / BLOCK_SIZE), 1);

  // Allocate memory
  uchar* data_gpu;
  checkCudaErrors(cudaMalloc(&data_gpu, PIXELDIM3 * sizeof(uchar)));
  checkCudaErrors(cudaMemcpy(data_gpu, data, PIXELDIM3 * sizeof(uchar), cudaMemcpyHostToDevice));

  NewtonGPUCalc<<<DimGrid, DimBlock>>>(data_gpu);
  checkCudaErrors(cudaGetLastError());

  cudaDeviceSynchronize();

  checkCudaErrors(cudaMemcpy(data, data_gpu, PIXELDIM3 * sizeof(uchar), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(data_gpu));
}
//#include "cuda_runtime.h"
//#include <helper_cuda.h>
//#include <device_launch_parameters.h>
//#include "MandrelBrot.h"
//
//#define WIDTH 512
//#define BLOCK 16
//#define max_iterations 1000
//
//__device__ int mandrelbrotset(float x, float y, int iterations)
//{
//	float newx = 0.0f, newy = 0.0f, xsq = 0.0f, ysq = 0.0f;
//	int itr = 0;
//
//	while ((xsq + ysq) < 4 && (itr < iterations))
//	{
//		newy = 2 * newx*newy + y;
//		newx = xsq - ysq + x;
//		xsq = newx*newx;
//		ysq = newy*newy;
//		itr++;
//	}
//	return itr;
//}
//
//
//__global__ void Kernel(uchar* data, int iterationMax)
//{
//	float step = 0.005f;
//	int XID = blockDim.x + blockIdx.x + threadIdx.x;
//	int YID = blockDim.y*blockIdx.y + threadIdx.y;
//	int iter;
//	float minx = -2.1f;
//	float maxx = 0.5f;
//	float miny = -1.3f;
//	float maxy = 1.3f;
//	int location = XID;
//
//	for (int j = YID + miny; j < maxy; j += blockDim.y*gridDim.y*step) 
//	{
//		for (int i = XID + minx; i < maxx; i += blockDim.x*gridDim.x*step) 
//		{
//			iter = mandrelbrotset(i, j, iterationMax - 1);
//
//			if (iter >= 999)
//			{
//				data[location] = 0;
//				data[location + 1] = 0;
//				data[location + 2] = 0;
//			}
//			else
//			{
//				data[location] = 0x18;
//				data[location + 1] = 0xff;
//				data[location + 2] = 0xad;
//
//			}
//			location += XID;
//
//		}
//	}
//}
//
////extern "C" void MandreGPU(uchar* data)
//{
//	cudaMalloc((void**)&data, sizeof(uchar*)*(WIDTH*WIDTH));
//	dim3 grids(WIDTH / BLOCK, WIDTH, BLOCK);
//	dim3 threads(BLOCK, BLOCK);
//	Kernel<<<grids, threads>>>(data, max_iterations);
//}
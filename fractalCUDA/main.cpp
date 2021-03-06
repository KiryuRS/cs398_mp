/******************************************************************************/
/*!
@file   Main.cpp
@par    Purpose: Main file of the program
@par    Language: C++
@par    Platform: Visual Studio 2015, Windows 10 64-bit
@author Kenneth, Alvin, YongKiat, ChengJiang
@par    Email:	t.weigangkenneth\@digipen.edu, alvin.tan\@digipen.edu,
				yongkiat.ong\@digipen.edu, chengjiang.tham\@digipen.edu
@date   07/12/2018
*/
/******************************************************************************/

//#define YONGKIAT_MandrelVERSION
//#define YONGKIAT_TriangleVERSION
//#define ALVIN_Henon
//#define ALVIN_Newton
//#define ALVIN_Ikeda
#define CHENGJIANG_VERSION_BurningShip
//#define CHENGJIANG_VERSION_FractalTree
//#define KENNETH_VERSION_BROWNIANTREE
//#define KENNETH_VERSION_SIERPINSKICARPET

#include "Common.h"

void PrintInformation(cudaDeviceProp& deviceProp)
{
	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
		driverVersion / 1000, (driverVersion % 100) / 10,
		runtimeVersion / 1000, (runtimeVersion % 100) / 10);

	printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
		deviceProp.major, deviceProp.minor);

	printf("  Total amount of global memory:                 %.2f GBytes (%llu "
		"bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
		(unsigned long long)deviceProp.totalGlobalMem);

	printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
		"GHz)\n", deviceProp.clockRate * 1e-3f,
		deviceProp.clockRate * 1e-6f);

	printf("  Memory Clock rate:                             %.0f Mhz\n",
		deviceProp.memoryClockRate * 1e-3f);

	printf("  Memory Bus Width:                              %d-bit\n",
		deviceProp.memoryBusWidth);

	if (deviceProp.l2CacheSize)
	{
		printf("  L2 Cache Size:                                 %d bytes\n",
			deviceProp.l2CacheSize);
	}

	printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
		"2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
		deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
		deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
		deviceProp.maxTexture3D[2]);

	printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
		"2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
		deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
		deviceProp.maxTexture2DLayered[1],
		deviceProp.maxTexture2DLayered[2]);

	printf("  Total amount of constant memory:               %zu bytes\n",
		deviceProp.totalConstMem);

	printf("  Total amount of shared memory per block:       %zu bytes\n",
		deviceProp.sharedMemPerBlock);

	printf("  Total number of registers available per block: %d\n",
		deviceProp.regsPerBlock);

	printf("  Warp size:                                     %d\n",
		deviceProp.warpSize);

	printf("  Maximum number of threads per multiprocessor:  %d\n",
		deviceProp.maxThreadsPerMultiProcessor);

	printf("  Maximum number of threads per block:           %d\n",
		deviceProp.maxThreadsPerBlock);

	printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
		deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);

	printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
		deviceProp.maxGridSize[0],
		deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);

	printf("  Maximum memory pitch:                          %zu bytes\n",
		deviceProp.memPitch);
}

int main(int argc, char **argv)
{
    // Declaring all the variables
    StopWatchInterface  *hTimer = nullptr;
    cudaDeviceProp       deviceProp;
    bmp_header           header;
	uchar               *cpuOutput = nullptr;
	uchar               *gpuOutput = nullptr;
	uchar               *cpuOutputPtr = nullptr;
	uchar               *gpuOutputPtr = nullptr;

    // Printing the information
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev = findCudaDevice(argc, (const char**)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
    deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
    PrintInformation(deviceProp);
	cudaDeviceSynchronize();

    // Setup bmp_header
	bmp_read("blank_bmp.bmp", &header, &cpuOutputPtr);
	size_t size = header.width * header.height * 3 * sizeof(uchar);
	header.h_resolution = 8192;
	header.v_resolution = 8192;

	sdkCreateTimer(&hTimer);

	// Some variable naming here
#ifdef YONGKIAT_MandrelVERSION

	Mandrelbrot man;
	const std::string fileOut{ "_YONGKIAT_Mandrel" };

#elif defined YONGKIAT_TriangleVERSION

	const std::string fileOut{ "_YONGKIAT_Triangle" };
	STriangle tr;

#elif defined ALVIN_Henon

	const std::string fileOut{ "_ALVIN_Henon" };

#elif defined ALVIN_Newton

	const std::string fileOut{ "_ALVIN_Newton" };
	bmp_read("blank_bmp.bmp", &header, &gpuOutputPtr);

#elif defined ALVIN_Ikeda

	const std::string fileOut{ "_ALVIN_Ikeda" };
	bmp_read("blank_bmp.bmp", &header, &gpuOutputPtr);

#elif defined CHENGJIANG_VERSION_BurningShip

	const std::string fileOut{ "_CHENGJIANG_BurningShip" };
	BurningShip ship;

#elif defined CHENGJIANG_VERSION_FractalTree

	const std::string fileOut{ "_CHENGJIANG_FractalTree" };
	FractalTree tree;

#elif defined KENNETH_VERSION_BROWNIANTREE

	const std::string fileOut{ "_KENNETH_BrownianTree" };

#elif defined KENNETH_VERSION_SIERPINSKICARPET

	const std::string fileOut{ "_KENNETH_Carpet" };

#endif

	// CPU CODE HERE
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// We will run the relevant code based on whose technique it is
	// SIGNATURE FOR FUNCTION CALL
	// void FuncName(cpuOutput, gpuOutput)
#ifdef YONGKIAT_MandrelVERSION

	man.MandrelbrotCPU(cpuOutputPtr);

#elif defined YONGKIAT_TriangleVERSION

	tr.TriangleCPU(cpuOutputPtr);

#elif defined ALVIN_Henon

	HenonCPU(cpuOutputPtr);

#elif defined ALVIN_Newton

	NewtonCPU(cpuOutputPtr);

#elif defined ALVIN_Ikeda

    IkedaGPU(cpuOutputPtr);

#elif defined CHENGJIANG_VERSION_BurningShip

	ship.BurningShipCPU(cpuOutputPtr);

#elif defined CHENGJIANG_VERSION_FractalTree

	tree.FractalTreeCPU(cpuOutputPtr);

#elif defined KENNETH_VERSION_BROWNIANTREE

	cpuOutput = new uchar[size];
	MyCopy(cpuOutputPtr, cpuOutputPtr + size, cpuOutput);
	BrownianCPU(cpuOutputPtr);

#elif defined KENNETH_VERSION_SIERPINSKICARPET

	SierpinskiCarpetCPU(nullptr);

#endif

	uchar * simulateCPUmallocTime = (uchar *)malloc(PIXELDIM3 * sizeof(uchar));
	std::memset(simulateCPUmallocTime, 0x00, PIXELDIM3 * sizeof(uchar));
	sdkStopTimer(&hTimer);
	double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	printf("\nCPU time (average) : %.5f sec, %.4f MB/sec\n", dAvgSecs, ((double)PIXELDIM3 * 1.0e-6) / dAvgSecs);
	printf("CPU Fractal, Throughput = %.4f MB/s, Time = %.5f s, Size = %zu Bytes, NumDevsUsed = %u\n",
		(1.0e-6 * (double)PIXELDIM3 / dAvgSecs), dAvgSecs, PIXELDIM3, 1u);
	free(simulateCPUmallocTime);


	// GPU  CODE HERE
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// We will run the relevant code based on whose technique it is
	// SIGNATURE FOR FUNCTION CALL
	// void FuncName(cpuOutput, gpuOutput)
#ifdef YONGKIAT_MandrelVERSION

	man.MandrelbrotGPU(&gpuOutputPtr);

#elif defined YONGKIAT_TriangleVERSION

	tr.TriangleGPU(&gpuOutputPtr);

#elif defined Alvin_Henon



#elif defined ALVIN_Newton

	NewtonGPU(gpuOutputPtr);

#elif defined ALVIN_Ikeda

    IkedaGPU(gpuOutputPtr);

#elif defined CHENGJIANG_VERSION_BurningShip

	ship.BurningShipGPU(&gpuOutputPtr);

#elif defined CHENGJIANG_VERSION_FractalTree

	tree.FractalTreeGPU(&gpuOutputPtr);

#elif defined KENNETH_VERSION_BROWNIANTREE

	BrownianGPU(cpuOutput, &gpuOutputPtr);

#elif defined KENNETH_VERSION_SIERPINSKICARPET

	SierpinskiCarpetGPU(nullptr, nullptr);

#endif

	sdkStopTimer(&hTimer);
	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	printf("\nGPU time (average) : %.5f sec, %.4f MB/sec\n", dAvgSecs, ((double)PIXELDIM3 * 1.0e-6) / dAvgSecs);
	printf("GPU Fractal, Throughput = %.4f MB/s, Time = %.5f s, Size = %zu Bytes, NumDevsUsed = %u\n",
		(1.0e-6 * (double)PIXELDIM3 / dAvgSecs), dAvgSecs, PIXELDIM3, 1u);
	sdkDeleteTimer(&hTimer);

#pragma region COPYOUT

	// Copying the contents over to our output file
	std::string cpuOutputFile{ "cpuOutput" };
	cpuOutputFile += fileOut + ".bmp";

	if (cpuOutputPtr)
		bmp_write((char*)cpuOutputFile.c_str(), &header, cpuOutputPtr);
	std::string gpuOutputFile{ "gpuOutput" };
	gpuOutputFile += fileOut + ".bmp";

	if (gpuOutputPtr)
		bmp_write((char*)gpuOutputFile.c_str(), &header, gpuOutputPtr);

	delete[] cpuOutputPtr;

#pragma endregion

#if not defined KENNETH_VERSION_BROWNIANTREE || not defined KENNETH_VERSION_SIERPINSKICARPET
	std::string command;
	command = "start WinMerge " + gpuOutputFile + " " + cpuOutputFile;
	system(command.c_str());
#elif defined KENNETH_VERSION_SIERPINSKICARPET
	std::string command;
	command = "start WinMerge gpuOutput_KENNETH_Carpet.txt cpuOutput_KENNETH_Carpet.txt";
	system(command.c_str());
#endif

	// For deallocating memory for GPUOutput (GPU side)
#ifdef YONGKIAT_MandrelVERSION

	man.ClearMemory(&gpuOutputPtr);

#elif defined YONGKIAT_TriangleVERSION

	tr.ClearMemory(&gpuOutputPtr);

#elif defined ALVIN_Henon



#elif defined ALVIN_Newton | defined ALVIN_Ikeda

	delete[] gpuOutputPtr;

#elif defined CHENGJIANG_VERSION_BurningShip

	ship.clearGPUMemory(&gpuOutputPtr);

#elif defined CHENGJIANG_VERSION_FractalTree

	tree.clearGPUMemory(&gpuOutputPtr);

#elif defined KENNETH_VERSION_BROWNIANTREE

	delete[] cpuOutput;
	BrownianClearGPU(&gpuOutputPtr);

#elif defined KENNETH_VERSION_SIERPINSKICARPET

	SierpinskiCarpetFree();

#endif
	return 0;
}
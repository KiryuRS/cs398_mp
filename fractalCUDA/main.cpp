// @TODO: Uncomment the codes below to run the relevant test case
//#define YONGKIAT_VERSION
//#define ALVIN_VERSION
//#define CHENGJIANG_VERSION
#define KENNETH_VERSION

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
	StopWatchInterface	*hTimer = nullptr;
	cudaDeviceProp		deviceProp;
	bmp_header			header;
	uchar				cpuOutput[PIXELDIM3]{ };
    uchar               *cpuOutputPtr;
	uchar				*gpuOutput = nullptr;


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
    //header.id1 = 'B';
    //header.id2 = 'M';
    //header.file_size = PIXELDIM3;
    //header.reserved = 0;
    //header.bmp_data_offset = 54;
    //header.bmp_header_size = sizeof(bmp_header);
    //header.width    = PIXELDIM;
    //header.height   = PIXELDIM;
    //header.planes = 1;
    //header.bits_per_pixel = 24;
    //header.h_resolution = 3800;
    //header.v_resolution = 3800;
    //header.colors = 0;
    //header.important_colors = 0;

    bmp_read("blank_bmp.bmp", &header, &cpuOutputPtr);
	size_t size = header.width * header.height * 3 * sizeof(uchar);
	MyCopy(cpuOutputPtr, cpuOutputPtr + size, cpuOutput);
    header.h_resolution = 8192;
    header.v_resolution = 8192;

	sdkCreateTimer(&hTimer);

	// Some variable naming here
#ifdef YONGKIAT_VERSION
	const std::string fileOut{ "_YONGKIAT" };
#elif defined ALVIN_VERSION
	const std::string fileOut{ "_ALVIN" };
#elif defined CHENGJIANG_VERSION
	const std::string fileOut{ "_CHENGJIANG" };

	BurningShip ship;

#elif defined KENNETH_VERSION
	const std::string fileOut{ "_KENNETH" };
#endif

	// CPU CODE HERE
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// We will run the relevant code based on whose technique it is
	// SIGNATURE FOR FUNCTION CALL
	// void FuncName(cpuOutput, gpuOutput)
#ifdef YONGKIAT_VERSION
	TriangleCPU(cpuOutputPtr);
	//MandrelbrotCPU(cpuOutputPtr);
	
#elif defined ALVIN_VERSION
    // HenonCPU(cpuOutputPtr);
    NewtonCPU(cpuOutputPtr);
#elif defined CHENGJIANG_VERSION
	ship.BurningShipCPU(cpuOutputPtr);
	//FractalTreeCPU(cpuOutputPtr);
#elif defined KENNETH_VERSION
	BrownianCPU(cpuOutputPtr);
#endif
	sdkStopTimer(&hTimer);
	double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	printf("\nCPU time (average) : %.5f sec, %.4f MB/sec\n", dAvgSecs, ((double)PIXELDIM3 * 1.0e-6) / dAvgSecs);
	printf("CPU Fractal, Throughput = %.4f MB/s, Time = %.5f s, Size = %zu Bytes, NumDevsUsed = %u\n",
		(1.0e-6 * (double)PIXELDIM3 / dAvgSecs), dAvgSecs, PIXELDIM3, 1u);

	// GPU  CODE HERE
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// We will run the relevant code based on whose technique it is
	// SIGNATURE FOR FUNCTION CALL
	// void FuncName(cpuOutput, gpuOutput)
#ifdef YONGKIAT_VERSION
	//MandrelbrotGPU(gpuOutput);
#elif defined ALVIN_VERSION

#elif defined CHENGJIANG_VERSION
	ship.BurningShipGPU(&gpuOutput);
#elif defined KENNETH_VERSION
	BrownianGPU(cpuOutput, &gpuOutput);
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
	char* out = new char[PIXELDIM3]{ };
	bmp_write((char*)cpuOutputFile.c_str(), &header, cpuOutputPtr);

	std::string gpuOutputFile{ "gpuOutput" };
	gpuOutputFile += fileOut + ".bmp";
	if (!gpuOutput)
	{
		std::cerr << "..Please allocate memory for gpuOutput!\nUnable to output file into a bmp file!" << std::endl;
		return -1;
	}
	MyCopy(gpuOutputFile.begin(), gpuOutputFile.end(), out);
	bmp_write(out, &header, gpuOutput);
	delete[] cpuOutputPtr;
	delete[] out;
#pragma endregion

	// std::string command;
	// command = "start WinMerge " + gpuOutputFile + " " + cpuOutputFile;
	// system(command.c_str());

	// For deallocating memory for GPUOutput (GPU side)
#ifdef YONGKIAT_VERSION

#elif defined ALVIN_VERSION

#elif defined CHENGJIANG_VERSION
	ship.clearGPUMemory(&gpuOutput);
#elif defined KENNETH_VERSION

#endif

	return 0;
}

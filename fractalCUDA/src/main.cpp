#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

//add into Project/Properties/CUDA C/C++ Additional Include Directories
//C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.2\common\inc;
// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include "heat.h"
//#include "kernel.cu"

// project include
#include <stdint.h>

#define epsilon 1.0e-3 

const static char *sSDKsample = "[heat distribution]\0";


enum file_name : uint
{
	YongKiat = 0,
	Kenneth,
	Alvin,
	ChengJiang

};

int main(int argc, char **argv)
{
	StopWatchInterface *hTimer = NULL;
	int PassFailFlag = 1;
	uint project;

	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	if (argc < 2) {
		printf
		("Please Pick Fractals by Input Data\t :\n\nMandelBrot\t\t\t\t : 0 0\nKoch SnowFlake\t\t\t\t : 0 1\nSelf Avoiding Walks\t\t\t : 1 0\nBrownian Tree\t\t\t\t : 1 1\nBurning Ship Fractals\t\t\t : 2 0\nFractal Tree\t\t\t\t : 2 1\nAttractors\t\t\t\t : 3 0\nSierpinski Carpet\t\t\t : 3 1\n\n");
		exit(0);
	}

	uint name = atoi(argv[1]);
	if (name > 3)
	{
		printf("0 : Yong Kiat, 1 : Kenneth, 2 : Alvin, 3 : Cheng Jiang");
		exit(0);
	}

	if (argc == 2)
		project = 0;

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	sdkCreateTimer(&hTimer);

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	switch (name)
	{
	case YongKiat :


		switch (project)
		{
		case 0 :

			MandelBrotGPU();




			break;
		case 1 :
			KochSnowflakeGPU();



			break;
		}



		break;
	case Kenneth :


		switch (project)
		{
		case 0:


			SelfAvoidingWalksGPU();



			break;
		case 1:


			BrownianTreeGPU();


			break;
		}



		break;
	case Alvin :

		switch (project)
		{
		case 0:


			AttractorsGPU();



			break;
		case 1:


			SierpinskiCarpetGPU();



			break;
		}




		break;
	case ChengJiang :

		switch (project)
		{
		case 0:

			BurningShipFractalGPU();





			break;
		case 1:


			FractalTreeGPU();



			break;
		}





		break;
	}



	sdkStopTimer(&hTimer);

	float dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) ;
	printf("GPU version time (average) : %.5f sec\n\n", dAvgSecs);
	//printf("GPU version time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)(count * sizeof(float)) * 1.0e-6) / dAvgSecs);
	//printf("GPU version , Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u\n",
		//(1.0e-6 * (double)(count*sizeof(float)) / dAvgSecs), dAvgSecs, (count*sizeof(float)), 1);

	printf("Shutting down...\n");


	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();


	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	switch (name)
	{
	case YongKiat:


		switch (project)
		{
		case 0:

			MandelBrotCPU();




			break;
		case 1:

			KochSnowflakeCPU();







			break;
		}



		break;
	case Kenneth:


		switch (project)
		{
		case 0:


			SelfAvoidingWalksCPU();


			break;
		case 1:

			BrownianTreeCPU();



			break;
		}



		break;
	case Alvin:

		switch (project)
		{
		case 0:

			AttractorsCPU();





			break;
		case 1:

			SierpinskiCarpetCPU();




			break;
		}




		break;
	case ChengJiang:

		switch (project)
		{
		case 0:

			BurningShipFractalCPU();





			break;
		case 1:


			FractalTreeCPU();




			break;
		}





		break;
	}




	sdkStopTimer(&hTimer);
	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer); // (double)numRuns;
	printf("CPU version time (average) : %.5f sec\n\n", dAvgSecs);
	//printf("CPU version time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)(count * sizeof(float))  * 1.0e-6) / dAvgSecs);
	//printf("CPU version, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
	//	(1.0e-6 * (double)(count*sizeof(float)) / dAvgSecs), dAvgSecs, count * sizeof(float));

	printf("Shutting down...\n");
	sdkDeleteTimer(&hTimer);

	//printf(" ...comparing the results\n");
//#define DEBUG
#ifdef DEBUG
	for (uint i = 0; i < nRowPoints; i++) {
		for (uint j = 0; j < nRowPoints; j++) {
			printf("%.2f ", *(h_DataGPU + i*nRowPoints + j));
		}
		printf("\n");
	}
#endif
	//PassFailFlag = 1;
	//for (uint i = 0; i < count; i++) 
	//	if (abs(h_DataGPU[i] - h_DataOutCPU[i]) > epsilon )
	//	{
	//		printf("%d %f %f\n", i, h_DataGPU[i], h_DataOutCPU[i]);
	//		PassFailFlag = 0;
	//		break;
	//	}
	//
	//printf(PassFailFlag ? " ...results match\n\n" : " ***results do not match!!!***\n\n");
	//
	//
	//printf("%s - Test Summary\n", sSDKsample);
#if 0
	// pass or fail (for both 64 bit and 256 bit histograms)
	if (!PassFailFlag)
	{
		printf("Test failed!\n");
		return -1;
	}

	printf("Test passed\n");
#endif

	return 0;
}

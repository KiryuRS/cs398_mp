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

int main(int argc, char **argv)
{
	float *d_DataIn;
	float *d_DataOut;
	StopWatchInterface *hTimer = NULL;
	int PassFailFlag = 1;
	uint count;
	uint nIter;

	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	if (argc != 3 ) {
		printf
		("Usage: heat nRowPoints nIter \n\n Total number of points = nRowPoints*nRowPoints\n\n");
		exit(0);
	}
	
	nIter = (uint) atoi(argv[2]);
	printf("%d Iterations\n", nIter);

	// set logfile name and start logs
	printf("[%s] - Starting...\n", sSDKsample);

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	sdkCreateTimer(&hTimer);
	printf("...allocating CPU memory.\n");

	printf("Initializing data...\n");
	printf("...reading input data\n");
	uint nRowPoints = atoi(argv[1]);
	count = nRowPoints*nRowPoints;
	float *h_DataGPU = (float *)malloc(count*sizeof(float));

	printf("...allocating GPU memory and copying input data\n\n");
	checkCudaErrors(cudaMalloc((void **)&d_DataIn, count*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_DataOut, count*sizeof(float)));

	initPoints(h_DataGPU, h_DataGPU, nRowPoints);
	checkCudaErrors(cudaMemcpy(d_DataIn, h_DataGPU, count*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_DataOut, h_DataGPU, count*sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	heatDistrGPU(d_DataIn, d_DataOut, nRowPoints, nIter);

//	printf("\nValidating GPU results...\n");
//	printf(" ...reading back GPU results\n");
	checkCudaErrors(cudaMemcpy(h_DataGPU, d_DataIn, count*sizeof(float), cudaMemcpyDeviceToHost));

	sdkStopTimer(&hTimer);

	float dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) ;
	printf("GPU version time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)(count * sizeof(float)) * 1.0e-6) / dAvgSecs);
	printf("GPU version , Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u\n",
		(1.0e-6 * (double)(count*sizeof(float)) / dAvgSecs), dAvgSecs, (count*sizeof(float)), 1);

	printf("Shutting down...\n");
	checkCudaErrors(cudaFree(d_DataIn));
	checkCudaErrors(cudaFree(d_DataOut));

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	float* h_DataInCPU = (float*) malloc(count * sizeof(float));
	float* h_DataOutCPU = (float*) malloc(count * sizeof(float));

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	initPoints(h_DataInCPU, h_DataOutCPU, nRowPoints);
	heatDistrCPU(	h_DataInCPU, h_DataOutCPU, nRowPoints, nIter);

	sdkStopTimer(&hTimer);
	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer); // (double)numRuns;
	printf("CPU version time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)(count * sizeof(float))  * 1.0e-6) / dAvgSecs);
	printf("CPU version, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
		(1.0e-6 * (double)(count*sizeof(float)) / dAvgSecs), dAvgSecs, count * sizeof(float));

	printf("Shutting down...\n");
	sdkDeleteTimer(&hTimer);

	printf(" ...comparing the results\n");
//#define DEBUG
#ifdef DEBUG
	for (uint i = 0; i < nRowPoints; i++) {
		for (uint j = 0; j < nRowPoints; j++) {
			printf("%.2f ", *(h_DataGPU + i*nRowPoints + j));
		}
		printf("\n");
	}
#endif
	PassFailFlag = 1;
	for (uint i = 0; i < count; i++) 
		if (abs(h_DataGPU[i] - h_DataOutCPU[i]) > epsilon )
		{
			printf("%d %f %f\n", i, h_DataGPU[i], h_DataOutCPU[i]);
			PassFailFlag = 0;
			break;
		}

	printf(PassFailFlag ? " ...results match\n\n" : " ***results do not match!!!***\n\n");

	free(h_DataGPU);
	free(h_DataInCPU);
	free(h_DataOutCPU);

	printf("%s - Test Summary\n", sSDKsample);
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

#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"
#include "SSFFT_GPU_base.cuh"
// #include "SSFFT_GPU_1SM.cuh"
// #include "SSFFT_GPU_2SMs.cuh"
// #include "SSFFT_GPU_6SMs.cuh"


#define WARP 32

int device=0;

__device__ __inline__ float shfl(float *value, int par){
	#if (CUDART_VERSION >= 9000)
		return(__shfl_sync(0xffffffff, (*value), par));
	#else
		return(__shfl((*value), par));
	#endif
}

void FFT_init(){
	//---------> Specific nVidia stuff
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}

// ***********************************************************************************
int GPU_cuFFT(float2 *h_input, float2 *h_output, int FFT_size, int nFFTs, int nRuns, double *single_ex_time){
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem,total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(devCount>device) checkCudaErrors(cudaSetDevice(device));
	
	//---------> Checking memory
	cudaMemGetInfo(&free_mem,&total_mem);
	if(DEBUG) printf("\n  Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float) total_mem)/(1024.0*1024.0), (float) free_mem/(1024.0*1024.0));
	size_t input_size = FFT_size*nFFTs;
	size_t output_size = FFT_size*nFFTs;
	size_t total_memory_required_bytes = input_size*sizeof(float2) + output_size*sizeof(float2);
	if(total_memory_required_bytes>free_mem) {
		printf("Error: Not enough memory! Input data are too big for the device.\n");
		return(1);
	}
	
	//----------> Memory allocation
	float2 *d_input;
	float2 *d_output;
	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(float2)*input_size));
	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
	
	//---------> Measurements
	double time_cuFFT = 0;
	GpuTimer timer;
		
	//--------------------------------------------------
	//-------------------------> cuFFT
	cufftHandle plan;
	cufftResult error;
	// error = cufftPlan1d(&plan, FFT_size, CUFFT_C2C, nFFTs);
    error = cufftPlan1d(&plan, FFT_size, CUFFT_C2C, nFFTs);
	if (CUFFT_SUCCESS != error){
		printf("CUFFT error: %d", error);
	}
	
	for (int i = 0; i < nRuns; i++) {
		timer.Start();
        checkCudaErrors(cudaMemcpy(d_input, h_input, input_size*sizeof(float2), cudaMemcpyHostToDevice));
		cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_output, CUFFT_INVERSE);
		timer.Stop();
		if (i >= 10)
			time_cuFFT += timer.Elapsed();
	}
	
	
	checkCudaErrors(cudaMemcpy(h_output, d_output, output_size*sizeof(float2), cudaMemcpyDeviceToHost));
	
	cufftDestroy(plan);
	//-----------------------------------<
	//--------------------------------------------------
	
	printf("  FFT size: %d\n  cuFFT time = %0.3f ms\n", FFT_size, time_cuFFT/(nRuns-10));
	
	cudaDeviceSynchronize();
	
	//---------> Copy Device -> Host
	checkCudaErrors(cudaMemcpy(h_output, d_output, output_size*sizeof(float2), cudaMemcpyDeviceToHost));

	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	
	return(0);
}


int GPU_FFT_C2C_Stockham(float2 *h_input, float2 *h_output, int FFT_exp, int FFT_size, int nFFTs, int nRuns, int nSMs, double *single_ex_time){
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem,total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(devCount>device) checkCudaErrors(cudaSetDevice(device));
	
	//---------> Checking memory
	cudaMemGetInfo(&free_mem,&total_mem);
	if(DEBUG) printf("\n  Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float) total_mem)/(1024.0*1024.0), (float) free_mem/(1024.0*1024.0));
	size_t input_size = FFT_size*nFFTs;
	size_t output_size = FFT_size*nFFTs;
	size_t input_size_bytes  = FFT_size*nFFTs*sizeof(float2);
	size_t output_size_bytes = FFT_size*nFFTs*sizeof(float2);
	size_t total_memory_required_bytes = input_size*sizeof(float2) + output_size*sizeof(float2);
	if(total_memory_required_bytes>free_mem) {
		printf("Error: Not enough memory! Input data is too big for the device.\n");
		return(1);
	}
	
	//---------> Measurements
	double time_SSFFT = 0;
	GpuTimer timer; 
	
	//---------> Memory allocation
	float2 *d_output;
	float2 *d_input;
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input,  input_size_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_output, output_size_bytes));
	timer.Stop();

	// SSFFT
	if(SSFFT){
		if (DEBUG) printf("  Running SSFFT (Stockham)... ");
		FFT_init();
		double total_time_SSFFT = 0;
		for(int f=0; f<nRuns; f++){
			checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
			switch (nSMs) {
			// case 0:  // [baseline]
			// 	SSFFT_benchmark_base(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs, &total_time_SSFFT);
			// 	break;
			// case 0:  // [1 SM  / 1 FFT operation]
			// 	SSFFT_benchmark_1SM(d_input, d_output, FFT_exp, FFT_size, nFFTs, 1, &total_time_SSFFT);
			// 	break;
			// case 2:  // [2 SMs / 1 FFT operation]
			// 	SSFFT_benchmark_2SMs(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs, &total_time_SSFFT);
			// 	break; 
			// case 3:  // [3 SMs / 1 FFT operation]
			// 	SSFFT_benchmark_2SMs(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs, &total_time_SSFFT);
			// 	break;
			// case 6:  // [6 SMs / 1 FFT operation]
			// 	SSFFT_benchmark_6SMs(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs, &total_time_SSFFT);
			// 	break;
			// case 7:  // sequential execution (cuFFT ver.)
			// 	SSFFT_benchmark_base(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs, &total_time_SSFFT);
			// 	break;
			// case 8:  // parallel execution (6 kernels)
			// 	SSFFT_benchmark_base(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs, &total_time_SSFFT);
			// 	break;
			// case 9:  // parallel execution (3 kernels)
			// 	SSFFT_benchmark_2SMs(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs, &total_time_SSFFT);
			// 	break;
			// case 10:  // parallel execution (2 kernels)
			// 	SSFFT_benchmark_2SMs(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs, &total_time_SSFFT);
			// 	break;
			// case 11:  // sequential execution (scale-up)
			// 	SSFFT_benchmark_6SMs(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs, &total_time_SSFFT);
			// 	break;
			default: 
				// printf("# SMs error! [nSMs = 0, 1, 2, 3, 6, 7]\n");
				SSFFT_benchmark_base(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs, &total_time_SSFFT);
			}
			
			// Remove initialization overhead
			// if (f < 100) total_time_SSFFT = 0;
		}
        // cudaDeviceSynchronize();
		time_SSFFT = total_time_SSFFT/(nRuns);
        // printf("  SSFFT time = %0.3f ms\n", time_SSFFT);
		printf("%0.3f\n", time_SSFFT);

		if (DEBUG) printf("done in %g ms.\n", time_SSFFT);
		*single_ex_time = time_SSFFT;
	}
	
		
	//-----> Copy chunk of output data to host
	if (FFT_exp < 13) {
		checkCudaErrors(cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost));
	}
	else {
		checkCudaErrors(cudaMemcpy(h_output, d_input, output_size_bytes, cudaMemcpyDeviceToHost));
	}

	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	// std::cerr << cudaGetErrorString(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	
	// printf("%0.3f ", time_SSFFT);
	
	return(0);
}

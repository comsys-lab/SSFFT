#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"
#include "SSFFT_GPU_base.cuh"

__global__ void 
__launch_bounds__(1024, 2)
SSFFT_GPU_Short_base(float2 *d_input, float2 *d_output, int FFT_exp, int FFT_size, int nSMs){ // in-place
	float2 SA_DFT_value_even, SA_DFT_value_odd;
    float2 SB_DFT_value_even, SB_DFT_value_odd;
	float2 SA_ftemp2, SA_ftemp;
    float2 SB_ftemp2, SB_ftemp;
	float2 W, W_B;

    int r, j, k, PoT, PoTm1;

    extern __shared__ float2 s_input[];

	if (threadIdx.y == 0 && threadIdx.x + blockIdx.x*blockDim.x < FFT_size/4) {

    for (int i = 0; i < FFT_size / blockDim.x; i++) {
        s_input[blockDim.x*i + threadIdx.x] = d_input[blockDim.x*i + threadIdx.x + (blockIdx.x*FFT_size)];
    }
    __syncthreads();

    // for (i = 0; i < FFT_size / blockDim.x; i++) {
    //     temp[i] = d_input[blockDim.x*i + threadIdx.x + (blockIdx.x*FFT_size)];
    // }
    // __syncthreads();
	
	PoT=1;
	PoTm1=0;

    PoTm1=PoT;
    PoT=PoT<<1;

    j=threadIdx.x;

    SA_ftemp  = s_input[threadIdx.x + blockDim.x*0];
    SA_ftemp2 = s_input[threadIdx.x + blockDim.x*0 + (FFT_size >> 1)];
    SB_ftemp  = s_input[threadIdx.x + blockDim.x*1];
    SB_ftemp2 = s_input[threadIdx.x + blockDim.x*1 + (FFT_size >> 1)];
    SA_DFT_value_even.x = SA_ftemp.x + SA_ftemp2.x;
    SA_DFT_value_even.y = SA_ftemp.y + SA_ftemp2.y;
    SA_DFT_value_odd.x  = SA_ftemp.x - SA_ftemp2.x;
    SA_DFT_value_odd.y  = SA_ftemp.y - SA_ftemp2.y;
    SB_DFT_value_even.x = SB_ftemp.x + SB_ftemp2.x;
    SB_DFT_value_even.y = SB_ftemp.y + SB_ftemp2.y;
    SB_DFT_value_odd.x  = SB_ftemp.x - SB_ftemp2.x;
    SB_DFT_value_odd.y  = SB_ftemp.y - SB_ftemp2.y;
    __syncthreads();
    
    s_input[j*PoT + (blockDim.x << 1)*0 ]         = SA_DFT_value_even;
    s_input[j*PoT + PoTm1 + (blockDim.x << 1)*0]  = SA_DFT_value_odd;
    s_input[j*PoT + (blockDim.x << 1)*1]          = SB_DFT_value_even;
    s_input[j*PoT + PoTm1 + (blockDim.x << 1)*1]  = SB_DFT_value_odd;
    __syncthreads();

	// Calculate boundary that FFT step length exceeds blockDim
	// Start
	int boundary = blockDim.x;
	int cnt = 1;
	while (boundary) {
		boundary >>= 1;
		cnt++;
	}
	__syncthreads();
	
	for(r=2;r<cnt;r++){
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j=threadIdx.x>>(r-1);
		k=threadIdx.x & (PoTm1-1);
		
		W=Get_W_value(PoT,k);

        SA_ftemp  = s_input[threadIdx.x + blockDim.x*0];
        SA_ftemp2 = s_input[threadIdx.x + blockDim.x*0 + (FFT_size >> 1)];
        SB_ftemp  = s_input[threadIdx.x + blockDim.x*1];
        SB_ftemp2 = s_input[threadIdx.x + blockDim.x*1 + (FFT_size >> 1)];
        SA_DFT_value_even.x = SA_ftemp.x + W.x*SA_ftemp2.x - W.y*SA_ftemp2.y;
        SA_DFT_value_even.y = SA_ftemp.y + W.x*SA_ftemp2.y + W.y*SA_ftemp2.x;
        SA_DFT_value_odd.x  = SA_ftemp.x - W.x*SA_ftemp2.x + W.y*SA_ftemp2.y;
        SA_DFT_value_odd.y  = SA_ftemp.y - W.x*SA_ftemp2.y - W.y*SA_ftemp2.x;
        SB_DFT_value_even.x = SB_ftemp.x + W.x*SB_ftemp2.x - W.y*SB_ftemp2.y;
        SB_DFT_value_even.y = SB_ftemp.y + W.x*SB_ftemp2.y + W.y*SB_ftemp2.x;
        SB_DFT_value_odd.x  = SB_ftemp.x - W.x*SB_ftemp2.x + W.y*SB_ftemp2.y;
        SB_DFT_value_odd.y  = SB_ftemp.y - W.x*SB_ftemp2.y - W.y*SB_ftemp2.x;
        __syncthreads();
        
        s_input[j*PoT + k + (blockDim.x << 1)*0]         = SA_DFT_value_even;
        s_input[j*PoT + k + PoTm1 + (blockDim.x << 1)*0] = SA_DFT_value_odd;
        s_input[j*PoT + k + (blockDim.x << 1)*1]         = SB_DFT_value_even;
        s_input[j*PoT + k + PoTm1 + (blockDim.x << 1)*1] = SB_DFT_value_odd;
        __syncthreads();
	}
	
	for(r=cnt;r<=FFT_exp;r++){
		PoTm1=PoT;
		PoT=PoT<<1;

        // j = i>>(r-(cnt-1)); 
        // int m = (PoT-1)>>(cnt-1);
        // k = blockDim.x*(i & m);

        W   = Get_W_value(PoT, threadIdx.x);
        W_B = Get_W_value(PoT, blockDim.x + threadIdx.x);

        SA_ftemp  = s_input[threadIdx.x + blockDim.x*0];
        SA_ftemp2 = s_input[threadIdx.x + blockDim.x*0 + (FFT_size >> 1)];
        SB_ftemp  = s_input[threadIdx.x + blockDim.x*1];
        SB_ftemp2 = s_input[threadIdx.x + blockDim.x*1 + (FFT_size >> 1)];
        SA_DFT_value_even.x = SA_ftemp.x + W.x*SA_ftemp2.x - W.y*SA_ftemp2.y;
        SA_DFT_value_even.y = SA_ftemp.y + W.x*SA_ftemp2.y + W.y*SA_ftemp2.x;
        SA_DFT_value_odd.x  = SA_ftemp.x - W.x*SA_ftemp2.x + W.y*SA_ftemp2.y;
        SA_DFT_value_odd.y  = SA_ftemp.y - W.x*SA_ftemp2.y - W.y*SA_ftemp2.x;
        SB_DFT_value_even.x = SB_ftemp.x + W_B.x*SB_ftemp2.x - W_B.y*SB_ftemp2.y;
        SB_DFT_value_even.y = SB_ftemp.y + W_B.x*SB_ftemp2.y + W_B.y*SB_ftemp2.x;
        SB_DFT_value_odd.x  = SB_ftemp.x - W_B.x*SB_ftemp2.x + W_B.y*SB_ftemp2.y;
        SB_DFT_value_odd.y  = SB_ftemp.y - W_B.x*SB_ftemp2.y - W_B.y*SB_ftemp2.x;
        __syncthreads();
        
        s_input[threadIdx.x + 0*PoT + 0]                  = SA_DFT_value_even;
        s_input[threadIdx.x + 0*PoT + PoTm1 + 0]          = SA_DFT_value_odd;
        s_input[threadIdx.x + 0*PoT + blockDim.x]         = SB_DFT_value_even;
        s_input[threadIdx.x + 0*PoT + PoTm1 + blockDim.x] = SB_DFT_value_odd;
        __syncthreads();
        
	}

    for (int i = 0; i < FFT_size / blockDim.x; i++) {
        d_output[blockDim.x*i + threadIdx.x + (blockIdx.x*FFT_size)] = s_input[blockDim.x*i + threadIdx.x];
    }
    __syncthreads();
	}

	//-------> END
}

__global__ void 
__launch_bounds__(1024, 2)
SSFFT_GPU_Long_step1_base(float2 *d_input, float2 *d_output, int Na, int FFT_size, int Na_exp, int Nb_exp, int nFFTs, int nSMs){ // Na = nFFTs, Nb = FFT_size
	float2 SA_DFT_value_even, SA_DFT_value_odd;
	float2 SA_ftemp2, SA_ftemp;
	float2 W;

    int r, j, k, PoT, PoTm1;

    extern __shared__ float2 s_input[];

	// int Na = (int)1 << Na_exp;
	// int FFT_size = (int)1 << Nb_exp;

	// calculate Na Nb-point FFTs (column wise)
	for (int iter = 0; iter < (Na+(gridDim.x*blockDim.y-1))/(gridDim.x*blockDim.y); iter++) {

		if (blockIdx.x*blockDim.y + threadIdx.y + iter*blockDim.y*gridDim.x < Na) {

		for (int i = 0; i < FFT_size / blockDim.x; i++) {
			s_input[threadIdx.x + blockDim.x*i + threadIdx.y*FFT_size] = d_input[(Na*(FFT_size/2))*i + Na*threadIdx.x + threadIdx.y + blockDim.y*blockIdx.x + blockDim.y*gridDim.x*iter];
		}
		__syncthreads();
		
		PoT=1;
		PoTm1=0;

		PoTm1=PoT;
		PoT=PoT<<1;

		j=threadIdx.x;

		SA_ftemp  = s_input[threadIdx.x + threadIdx.y*FFT_size];
		SA_ftemp2 = s_input[threadIdx.x + (FFT_size >> 1) + threadIdx.y*FFT_size];
		SA_DFT_value_even.x = SA_ftemp.x + SA_ftemp2.x;
		SA_DFT_value_even.y = SA_ftemp.y + SA_ftemp2.y;
		SA_DFT_value_odd.x  = SA_ftemp.x - SA_ftemp2.x;
		SA_DFT_value_odd.y  = SA_ftemp.y - SA_ftemp2.y;
		__syncthreads();
		
		s_input[j*PoT + threadIdx.y*FFT_size]          = SA_DFT_value_even;
		s_input[j*PoT + PoTm1 + threadIdx.y*FFT_size]  = SA_DFT_value_odd;
		__syncthreads();

		for(r=2;r<=Nb_exp;r++){
			PoTm1=PoT;
			PoT=PoT<<1;
			
			j=threadIdx.x>>(r-1);
			k=threadIdx.x & (PoTm1-1);
			
			W=Get_W_value(PoT,k);

			SA_ftemp  = s_input[threadIdx.x + threadIdx.y*FFT_size];
			SA_ftemp2 = s_input[threadIdx.x + (FFT_size >> 1) + threadIdx.y*FFT_size];
			SA_DFT_value_even.x = SA_ftemp.x + W.x*SA_ftemp2.x - W.y*SA_ftemp2.y;
			SA_DFT_value_even.y = SA_ftemp.y + W.x*SA_ftemp2.y + W.y*SA_ftemp2.x;
			SA_DFT_value_odd.x  = SA_ftemp.x - W.x*SA_ftemp2.x + W.y*SA_ftemp2.y;
			SA_DFT_value_odd.y  = SA_ftemp.y - W.x*SA_ftemp2.y - W.y*SA_ftemp2.x;
			__syncthreads();
			
			s_input[j*PoT + k + threadIdx.y*FFT_size]         = SA_DFT_value_even;
			s_input[j*PoT + k + PoTm1 + threadIdx.y*FFT_size] = SA_DFT_value_odd;
			__syncthreads();
		}

		// element wise multiply twiddle factor
		for (int i = 0; i < FFT_size / blockDim.x; i++) {
			W=Get_W_value(FFT_size*Na, (threadIdx.x + blockDim.x*i)*(threadIdx.y + blockIdx.x*blockDim.y + blockDim.y*gridDim.x*iter));
			SA_ftemp = s_input[threadIdx.x + blockDim.x*i + threadIdx.y*FFT_size];
			SA_DFT_value_even.x = W.x*SA_ftemp.x - W.y*SA_ftemp.y;
			SA_DFT_value_even.y = W.x*SA_ftemp.y + W.y*SA_ftemp.x;
			__syncthreads();

			s_input[threadIdx.x + blockDim.x*i + threadIdx.y*FFT_size] = SA_DFT_value_even;
			__syncthreads();
		}

		// copy result to d_output
		for (int i = 0; i < FFT_size / blockDim.x; i++) {
			d_output[(Na*(FFT_size/2))*i + Na*threadIdx.x + threadIdx.y + blockDim.y*blockIdx.x + blockDim.y*gridDim.x*iter] 
			= s_input[threadIdx.x + blockDim.x*i + threadIdx.y*FFT_size];
		}
		__syncthreads();
		}
	}
}

__global__ void 
__launch_bounds__(1024, 2)
SSFFT_GPU_Long_step2_base(float2 *d_input, float2 *d_output, int FFT_size, int Nb, int Na_exp, int Nb_exp, int nFFTs, int nSMs){ // Na = nFFTs, Nb = FFT_size
	float2 SA_DFT_value_even, SA_DFT_value_odd;
	float2 SA_ftemp2, SA_ftemp;
	float2 W;

    int r, j, k, PoT, PoTm1;

    extern __shared__ float2 s_input[];

	// int Nb = (int)1 << Nb_exp;
	// int FFT_size = (int)1 << Na_exp;

	// calculate Nb Na-point FFTs (row wise)
	for (int iter = 0; iter < (Nb+(gridDim.x*blockDim.y-1))/(gridDim.x*blockDim.y); iter++) {

		if (blockIdx.x*blockDim.y + threadIdx.y + iter*blockDim.y*gridDim.x < Nb) {

		for (int i = 0; i < FFT_size / blockDim.x; i++) {
			s_input[threadIdx.x + blockDim.x*i + threadIdx.y*FFT_size] 
			= d_input[threadIdx.x + blockDim.x*i + threadIdx.y*FFT_size + blockDim.y*blockIdx.x*FFT_size + blockDim.y*gridDim.x*FFT_size*iter];
		}
		__syncthreads();
		
		PoT=1;
		PoTm1=0;

		PoTm1=PoT;
		PoT=PoT<<1;

		j=threadIdx.x;

		SA_ftemp  = s_input[threadIdx.x + threadIdx.y*FFT_size];
		SA_ftemp2 = s_input[threadIdx.x + (FFT_size >> 1) + threadIdx.y*FFT_size];
		SA_DFT_value_even.x = SA_ftemp.x + SA_ftemp2.x;
		SA_DFT_value_even.y = SA_ftemp.y + SA_ftemp2.y;
		SA_DFT_value_odd.x  = SA_ftemp.x - SA_ftemp2.x;
		SA_DFT_value_odd.y  = SA_ftemp.y - SA_ftemp2.y;
		__syncthreads();
		
		s_input[j*PoT + threadIdx.y*FFT_size]          = SA_DFT_value_even;
		s_input[j*PoT + PoTm1 + threadIdx.y*FFT_size]  = SA_DFT_value_odd;
		__syncthreads();

		for(r=2;r<=Na_exp;r++){
			PoTm1=PoT;
			PoT=PoT<<1;
			
			j=threadIdx.x>>(r-1);
			k=threadIdx.x & (PoTm1-1);
			
			W=Get_W_value(PoT,k);

			SA_ftemp  = s_input[threadIdx.x + threadIdx.y*FFT_size];
			SA_ftemp2 = s_input[threadIdx.x + (FFT_size >> 1) + threadIdx.y*FFT_size];
			SA_DFT_value_even.x = SA_ftemp.x + W.x*SA_ftemp2.x - W.y*SA_ftemp2.y;
			SA_DFT_value_even.y = SA_ftemp.y + W.x*SA_ftemp2.y + W.y*SA_ftemp2.x;
			SA_DFT_value_odd.x  = SA_ftemp.x - W.x*SA_ftemp2.x + W.y*SA_ftemp2.y;
			SA_DFT_value_odd.y  = SA_ftemp.y - W.x*SA_ftemp2.y - W.y*SA_ftemp2.x;
			__syncthreads();
			
			s_input[j*PoT + k + threadIdx.y*FFT_size]         = SA_DFT_value_even;
			s_input[j*PoT + k + PoTm1 + threadIdx.y*FFT_size] = SA_DFT_value_odd;
			__syncthreads();
		}

		// copy the result by trasposing result matrix
		for (int i = 0; i < FFT_size / blockDim.x; i++) {
			d_output[Nb*threadIdx.x + (Nb*(FFT_size/2))*i + threadIdx.y + blockDim.y*blockIdx.x + blockDim.y*gridDim.x*iter] 
			= s_input[blockDim.x*i + threadIdx.x + threadIdx.y*FFT_size];
		}
		__syncthreads();
		}
	}
}

void SSFFT_GPU_Short_base(float2 *d_input, float2 *d_output, int FFT_exp, int FFT_size, int nFFTs, int nSMs) {

	dim3 blockSize(FFT_size/4, 1024/(FFT_size/4));

	int parallel = 12/nSMs;

	float2 *d_input_iter;
	float2 *d_output_iter;

	cudaStream_t s[parallel];
	for (int i = 0; i < parallel; i++) cudaStreamCreate(&s[i]);

	for (int i = 0; i < (nFFTs+parallel-1)/parallel; i++) {
		d_input_iter = d_input + (FFT_size*i*parallel);
		d_output_iter = d_output + (FFT_size*i*parallel);

		for (int j = 0; j < parallel; j++) {
			if ((i*parallel + j) < nFFTs) {
				SSFFT_GPU_Short_base<<<nSMs, blockSize, 32768, s[j]>>>(d_input_iter+j*FFT_size, d_output_iter+j*FFT_size, FFT_exp, FFT_size, nSMs);
			}
		}
	}
	for (int i = 0; i < parallel; i++) cudaStreamDestroy(s[i]);
}


void SSFFT_GPU_Long_base(float2 *d_input, float2 *d_output, int FFT_exp, int FFT_size, int nFFTs, int nSMs) {

	int Na_exp = (FFT_exp/2);
	int Nb_exp = ((FFT_exp+1)/2);
	int Na = (int)1 << Na_exp;
	int Nb = (int)1 << Nb_exp;
	
	dim3 blockSize1(Nb/2, 1024/(Nb/2), 1);
	dim3 blockSize2(Na/2, 1024/(Na/2), 1);

	float2 *d_input_iter;
	float2 *d_output_iter;

	int parallel = 12/nSMs;

	cudaStream_t s[parallel];
	for (int i = 0; i < parallel; i++) cudaStreamCreate(&s[i]);

	for (int i = 0; i < (nFFTs+parallel-1)/parallel; i++) {
		d_input_iter = d_input + (FFT_size*i*parallel);
		d_output_iter = d_output + (FFT_size*i*parallel);

		for (int j = 0; j < parallel; j++) {
			if ((i*parallel + j) < nFFTs) {
				SSFFT_GPU_Long_step1_base<<<nSMs, blockSize1, 32768, s[j]>>>(d_input_iter+j*FFT_size, d_output_iter+j*FFT_size, Na, Nb, Na_exp, Nb_exp, nFFTs, nSMs); // Na_exp, Nb_exp
				SSFFT_GPU_Long_step2_base<<<nSMs, blockSize2, 32768, s[j]>>>(d_output_iter+j*FFT_size, d_input_iter+j*FFT_size, Na, Nb, Na_exp, Nb_exp, nFFTs, nSMs); // Na_exp, Nb_exp
			}
		}
	}
	for (int i = 0; i < parallel; i++) cudaStreamDestroy(s[i]);


	
}

// SSFFT_base
void SSFFT_benchmark_base(float2 *d_input, float2 *d_output, int FFT_exp, int FFT_size, int nFFTs, int nSMs, double *FFT_time){
	GpuTimer timer;
    timer.Start();

	if (FFT_size < 8192) {
		SSFFT_GPU_Short_base(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs);
	}	
	else {
		SSFFT_GPU_Long_base(d_input, d_output, FFT_exp, FFT_size, nFFTs, nSMs);
	}

    timer.Stop();
	*FFT_time += timer.Elapsed();
}
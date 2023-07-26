#ifndef SSFFT_GPU_BASE_H__
#define SSFFT_GPU_BASE_H__

__global__ void SSFFT_GPU_Short_base(float2 *d_input, float2 *d_output, int FFT_exp, int FFT_size);

__global__ void SSFFT_GPU_Long_step1_base(float2 *d_input, float2 *d_output, int Na_exp, int Nb_exp, int nFFTs);
__global__ void SSFFT_GPU_Long_step2_base(float2 *d_input, float2 *d_output, int Na_exp, int Nb_exp, int nFFTs);
void SSFFT_GPU_Long_base(float2 *d_input, float2 *d_output, int FFT_exp, int FFT_size, int nFFTs);

// SSFFT_base
void SSFFT_benchmark_base(float2 *d_input, float2 *d_output, int FFT_exp, int FFT_size, int nFFTs, int nSMs, double *FFT_time);

#endif
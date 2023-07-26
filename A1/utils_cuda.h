#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)


template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	//ctemp.x=-cosf( 6.283185f*fdividef( (float) m, (float) N ) - 3.141592654f );
	//ctemp.y=sinf( 6.283185f*fdividef( (float) m, (float) N ) - 3.141592654f );
	//ctemp.x=cosf( 2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	//ctemp.y=sinf( 2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	sincosf(6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
	return(ctemp);
}

#endif

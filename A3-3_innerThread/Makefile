INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda

# use this compilers
# g++ just because the file write
GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc


###############################################################
# Basic flags for compilers, one for debug options
# fmad flags used for reason of floating point operation
###############################################################
NVCCFLAGS = -O2 -gencode arch=compute_72,code=sm_72 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo 

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = SSFFT.exe

all: clean analyze

# analyze: SSFFT.o SSFFT_GPU.o SSFFT_GPU_base.o SSFFT_GPU_1SM.o SSFFT_GPU_2SMs.o SSFFT_GPU_6SMs.o Makefile
# 	$(NVCC) -o $(ANALYZE) SSFFT_GPU.o SSFFT_GPU_base.o SSFFT_GPU_1SM.o SSFFT_GPU_2SMs.o SSFFT_GPU_6SMs.o SSFFT.o $(LIB) $(NVCCFLAGS)

# SSFFT_GPU.o: timer.h utils_cuda.h SSFFT_GPU_base.cuh SSFFT_GPU_1SM.cuh SSFFT_GPU_2SMs.cuh SSFFT_GPU_6SMs.cuh
# 	$(NVCC) -c SSFFT_GPU.cu SSFFT_GPU_base.cu SSFFT_GPU_1SM.cu SSFFT_GPU_2SMs.cu SSFFT_GPU_6SMs.cu $(NVCCFLAGS)

analyze: SSFFT.o SSFFT_GPU.o SSFFT_GPU_base.o Makefile
	$(NVCC) -o $(ANALYZE) SSFFT_GPU.o SSFFT_GPU_base.o SSFFT.o $(LIB) $(NVCCFLAGS)

SSFFT_GPU.o: timer.h utils_cuda.h SSFFT_GPU_base.cuh 
	$(NVCC) -c SSFFT_GPU.cu SSFFT_GPU_base.cu $(NVCCFLAGS)

SSFFT.o: SSFFT.c
	$(GCC) -c SSFFT.c $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)



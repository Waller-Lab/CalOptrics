#ifndef FOURIER_H
#define FOURIER_H
 
#pragma once

#include <math.h>
#include "cuda_runtime.h"
#include <cuda.h>
#include <cufft.h> 
#include "device_launch_parameters.h"
#include "fourier.h"
#include "toolbox.h"

#define BLOCKSIZEX_FOURIER 16
#define BLOCKSIZEY_FOURIER 16

cudaError_t fft2D(cufftComplex *output, float *source, int height, int width);
cudaError_t ifft2D(float *output, cufftComplex *complex_source, int height, int width);
cudaError_t hostFftShift(cufftComplex *data, int height, int width);
cudaError_t hostIfftShift(float *data, int height, int width);

__global__ void fftShift1D(cufftComplex* data, int nX, int nY);
__global__ void ifftShift1D(float *data, int nX, int nY);

/**
 * @details
 * FFT Shift kernel.
 */
__global__
void fftShift(cufftComplex *data, int nX, int nY)
{
    
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.x + threadIdx.y;

    if (x < nX && y < nY) {
        float a = powf(-1.0, x + y);
        int index = y * nX + x;
        data[index].x *= a;
        data[index].y *= a;
    }
}

/**
 * @details
 * De-checkboard Real array after FFT kernel.
 */
__global__
void ifftShift(float *data, int nX, int nY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.x + threadIdx.y;

    if (x < nX && y < nY) {
        float a = powf(-1.0, x + y);
        int index = y * nX + x;
        data[index] *= a;
    }
}

// Helper function for using CUDA to perform 2D fft
cudaError_t fft2D(cufftComplex *output, float *source, int height, int width)
{
    cufftComplex *dev_output = 0;
    cufftComplex *dev_source = 0;
	cufftComplex *complex_source = toComplexArray(source, height*width);
    cudaError_t cudaStatus;
	cufftResult cufftStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for input and output.
    cudaStatus = cudaMalloc((void**)&dev_source, height*width * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_output, height*width * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffer.
    cudaStatus = cudaMemcpy(dev_source, complex_source, height * width * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Perform fft

	cufftHandle plan;

	cufftPlan2d(&plan, height, width, CUFFT_C2C);
	cufftStatus = cufftExecC2C(plan, dev_source, dev_output, CUFFT_FORWARD);
	
	if(cufftStatus != CUFFT_SUCCESS){
		fprintf(stderr, "cufftExecC2C returned error code %d after attempting 2D fft!\n", cufftStatus);
        goto Error;
	}
	

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, dev_output, height*width * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_output);
    cudaFree(dev_source);
	cufftDestroy(plan);
	free(complex_source);
    
    return cudaStatus;
}



// Helper function for using CUDA to perform 2D fft
cudaError_t ifft2D(float *real_output, cufftComplex *complex_source, int height, int width)
{
	cufftComplex *output = (cufftComplex *) malloc(sizeof(cufftComplex)*height*width);
	float *dev_real_output = 0;
    cufftComplex *dev_output = 0;
    cufftComplex *dev_source = 0;
    cudaError_t cudaStatus;
	cufftResult cufftStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for input and output.
    cudaStatus = cudaMalloc((void**)&dev_source, height*width * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_output, height*width * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_real_output, height*width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffer.
    cudaStatus = cudaMemcpy(dev_source, complex_source, height * width * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Perform ifft

	cufftHandle plan;

	cufftPlan2d(&plan, height, width, CUFFT_C2C);
	cufftStatus = cufftExecC2C(plan, dev_source, dev_output, CUFFT_INVERSE);
	
	if(cufftStatus != CUFFT_SUCCESS){
		fprintf(stderr, "cufftExecC2C returned error code %d after attempting 2D fft!\n", cufftStatus);
        goto Error;
	}
	
	
	dim3 myblock(BLOCKSIZEX_FOURIER,BLOCKSIZEY_FOURIER);
	dim3 mygrid( roundUpDiv(height,BLOCKSIZEX_FOURIER), roundUpDiv(width,BLOCKSIZEY_FOURIER) );
	float scale = 1.f / ( (float) height * (float) width );
	complex2real_scaled<<<mygrid, myblock>>>(dev_output, dev_real_output, height, width, scale);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(real_output, dev_real_output, height*width * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_output);
	cudaFree(dev_real_output);
    cudaFree(dev_source);
	cufftDestroy(plan);
	free(output);
    
    return cudaStatus;
}

cudaError_t hostFftShift(cufftComplex *source, int height, int width){
    cufftComplex *dev_source = 0;
    cudaError_t cudaStatus;
	int Nx = height;
	int Ny = width;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for input and output.
    cudaStatus = cudaMalloc((void**)&dev_source, height*width * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffer.
    cudaStatus = cudaMemcpy(dev_source, source, height * width * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_FOURIER;
    int block_size_y = BLOCKSIZEY_FOURIER;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	fftShift<<<dimGrid, dimBlock>>>(dev_source, Nx, Ny);	

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(source, dev_source, height*width * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_source);
    
    return cudaStatus;
}

cudaError_t hostIfftShift(float *source, int height, int width){
	float *dev_source = 0;
    cudaError_t cudaStatus;
	int Nx = height;
	int Ny = width;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for input and output.
    cudaStatus = cudaMalloc((void**)&dev_source, height*width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffer.
    cudaStatus = cudaMemcpy(dev_source, source, height * width * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_FOURIER;
    int block_size_y = BLOCKSIZEY_FOURIER;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	ifftShift<<<dimGrid, dimBlock>>>(dev_source, Nx, Ny);	

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(source, dev_source, height*width * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_source);
    
    return cudaStatus;
}

#endif
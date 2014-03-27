#ifndef POINTWISE_MATRIX_OPS_H
#define POINTWISE_MATRIX_OPS_H
 
#pragma once

#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include "toolbox.h"

#define BLOCKSIZEX_POINTWISE 16
#define BLOCKSIZEY_POINTWISE 16

/* device function declarations */

/* complete arithmetic function declarations */
__device__ cufftComplex add(cufftComplex a, cufftComplex b);
__device__ cufftComplex subtract(cufftComplex a, cufftComplex b);
__device__ cufftComplex multiply(cufftComplex a, cufftComplex b);
__device__ cufftComplex divide(cufftComplex a, cufftComplex b);
__device__ cufftComplex inverse(cufftComplex a);
__device__ cufftComplex scale(cufftComplex a, float b);
__device__ cufftComplex conjugate(cufftComplex a);
__device__ float magnitude(cufftComplex a);
__device__ float magnitudeSq(cufftComplex a);
/* complex matrices */
__global__ void pointwiseAddComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny);
__global__ void pointwiseSubtractComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny);
__global__ void pointwiseMultiplyComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny);
__global__ void pointwiseDivideComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny);
__global__ void pointwiseRealScaleComplexMatrix(cufftComplex *c, cufftComplex *a, float b, int Nx, int Ny);
__global__ void pointwiseComplexScaleComplexMatrix(cufftComplex *c, cufftComplex *a, cufftComplex b, int Nx, int Ny);
__global__ void pointwiseAddRealConstantToComplexMatrix(cufftComplex *c, cufftComplex *a, float b, int Nx, int Ny);
__global__ void pointwiseAddComplexConstantToComplexMatrix(cufftComplex *c, cufftComplex *a, cufftComplex b, int Nx, int Ny);
__global__ void pointwiseNaturalLogComplexMatrix(cufftComplex *c, cufftComplex *a, int Nx, int Ny);
__global__ void pointwisePowerComplexMatrix(cufftComplex *c, cufftComplex *a, int p,int Nx, int Ny);
/* real matrices */
__global__ void pointwiseAddRealMatrices(float *c, float *a, float *b, int Nx, int Ny);
__global__ void pointwiseSubtractRealMatrices(float *c, float *a, float *b, int Nx, int Ny);
__global__ void pointwiseMultiplyRealMatrices(float *c, float *a, float *b, int Nx, int Ny);
__global__ void pointwiseDivideRealMatrices(float *c, float *a, float *b, int Nx, int Ny);
__global__ void pointwiseRealScaleRealMatrix(float *c, float *a, float b, int Nx, int Ny);
__global__ void pointwiseAddRealConstantToRealMatrix(float *c, float *a, float b, int Nx, int Ny);
__global__ void pointwiseNaturalLogRealMatrix(float *c, float *a, int Nx, int Ny);
__global__ void pointwisePowerRealMatrix(float *c, float *a, int p,int Nx, int Ny);

/* host helper functions declarations*/

/* complex matrices */
cudaError_t hostPointwiseAddComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny);
cudaError_t hostPointwiseSubtractComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny);
cudaError_t hostPointwiseMultiplyComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny);
cudaError_t hostPointwiseDivideComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny);
cudaError_t hostPointwiseRealScaleComplexMatrix(cufftComplex *c, cufftComplex *a, float b, int Nx, int Ny);
cudaError_t hostPointwiseComplexScaleComplexMatrix(cufftComplex *c, cufftComplex *a, cufftComplex b, int Nx, int Ny);
cudaError_t hostPointwiseAddRealConstantToComplexMatrix(cufftComplex *c, cufftComplex *a, float b, int Nx, int Ny);
cudaError_t hostPointwiseAddComplexConstantToComplexMatrix(cufftComplex *c, cufftComplex *a, cufftComplex b, int Nx, int Ny);
cudaError_t hostPointwiseNaturalLogComplexMatrix(cufftComplex *c, cufftComplex *a, int Nx, int Ny);
cudaError_t hostPointwisePowerComplexMatrix(cufftComplex *c, cufftComplex *a, int p,int Nx, int Ny);

/* real matrices */
cudaError_t hostPointwiseAddRealMatrices(float *c, float *a, float *b, int Nx, int Ny);
cudaError_t hostPointwiseSubtractRealMatrices(float *c, float *a, float *b, int Nx, int Ny);
cudaError_t hostPointwiseMultiplyRealMatrices(float *c, float *a, float *b, int Nx, int Ny);
cudaError_t hostPointwiseDivideRealMatrices(float *c, float *a, float *b, int Nx, int Ny);
cudaError_t hostPointwiseRealScaleRealMatrix(float *c, float *a, float b, int Nx, int Ny);
cudaError_t hostPointwiseAddRealConstantToRealMatrix(float *c, float *a, float b, int Nx, int Ny);
cudaError_t hostPointwiseNaturalLogRealMatrix(float *c, float *a, int Nx, int Ny);
cudaError_t hostPointwisePowerRealMatrix(float *c, float *a, int p,int Nx, int Ny);

/* device functions */

/* complex matrices */
__global__ void pointwiseAddComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = add(a[index],b[index]);

    }
}

__global__ void pointwiseSubtractComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = subtract(a[index],b[index]);

    }
}

__global__ void pointwiseMultiplyComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny)
{
    /* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = multiply(a[index],b[index]);

    }
}

__global__ void pointwiseDivideComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny)
{
    /* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = divide(a[index],b[index]);

    }
}

__global__ void pointwiseRealScaleComplexMatrix(cufftComplex *c, cufftComplex *a, float b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = scale(a[index],b);
    }
}

__global__ void pointwiseComplexScaleComplexMatrix(cufftComplex *c, cufftComplex *a, cufftComplex b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = multiply(a[index],b);
    }
}

__global__ void pointwiseAddRealConstantToComplexMatrix(cufftComplex *c, cufftComplex *a, float b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index].x = a[index].x + b;
		c[index].y = a[index].y;
    }
}
__global__ void pointwiseAddComplexConstantToComplexMatrix(cufftComplex *c, cufftComplex *a, cufftComplex b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = add(a[index],b);
    }
}

__global__ void pointwiseNaturalLogComplexMatrix(cufftComplex *c, cufftComplex *a, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;
		
		if(a[index].x > 0){
			c[index].x = logf(a[index].x);
		}else {
			c[index].x = 0;
		}
        
		if(a[index].y > 0) {
			c[index].y = logf(a[index].y);
		}else {
			c[index].y = 0;
		}
    }
}

__global__ void pointwisePowerComplexMatrix(cufftComplex *c, cufftComplex *a, int exp, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;
		
		cufftComplex result;
		result.x = 1.0f;
		result.y = 0.0f;
		cufftComplex base = a[index];
		while (exp) {
			if (exp & 1)
				result = multiply(result, base);
			exp >>= 1;
			base = multiply(base, base);
		}

		c[index] = result;
    }
}

/* real matrices */
__global__ void pointwiseAddRealMatrices(float *c, float *a, float *b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = a[index] + b[index];

    }
}
__global__ void pointwiseSubtractRealMatrices(float *c, float *a, float *b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = a[index] - b[index];

    }
}
__global__ void pointwiseMultiplyRealMatrices(float *c, float *a, float *b, int Nx, int Ny) {
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = a[index] * b[index];

    }
}
__global__ void pointwiseDivideRealMatrices(float *c, float *a, float *b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = a[index] / b[index];

    }
}
__global__ void pointwiseRealScaleRealMatrix(float *c, float *a, float b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = a[index] * b;

    }
}
__global__ void pointwiseAddRealConstantToRealMatrix(float *c, float *a, float b, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = a[index] + b;

    }
}
__global__ void pointwiseNaturalLogRealMatrix(float *c, float *a, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;

        c[index] = logf(a[index]);

    }
}

__global__ void pointwisePowerRealMatrix(float *c, float *a, int exp, int Nx, int Ny){
	/* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;
		c[index] = powf(a[index], exp);
    }
}

/* complex number arithmetic functions */
__device__ cufftComplex add(cufftComplex a, cufftComplex b){
	cufftComplex cout;
	cout.x = a.x + b.x;
	cout.y = a.y + b.y;
	return cout;
}

__device__ cufftComplex subtract(cufftComplex a, cufftComplex b){
	cufftComplex cout;
	cout.x = a.x - b.x;
	cout.y = a.y - b.y;
	return cout;
}

__device__ cufftComplex multiply(cufftComplex a, cufftComplex b){
	cufftComplex cout;
	cout.x = a.x*b.x - a.y*b.y;
	cout.y = a.x*b.y + a.y*b.x;
	return cout;
}

__device__ cufftComplex divide(cufftComplex a, cufftComplex b){
	cufftComplex cout;
	cout = scale(multiply(a, conjugate(b)), 1/magnitudeSq(b));
	return cout;
}

__device__ cufftComplex inverse(cufftComplex a){
	cufftComplex cout;
	cufftComplex tmp;
	tmp.x = 1;
	tmp.y = 0;
	cout = divide(tmp, a);
	return cout;
}

__device__ cufftComplex conjugate(cufftComplex c){
	cufftComplex cout;
	cout.x = c.x;
	cout.y = -c.y;
	return cout;
}

__device__ float magnitude(cufftComplex c){
	return sqrtf(c.x*c.x + c.y*c.y);
}

__device__ float magnitudeSq(cufftComplex c){
	return c.x*c.x + c.y*c.y;
}

__device__ cufftComplex scale(cufftComplex a, float scale){
	cufftComplex cout;
	cout.x = a.x * scale;
	cout.y = a.y * scale;
	return cout;
}

/* host functions */

/* complex matrices */
cudaError_t hostPointwiseAddComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny){
	cufftComplex *dev_a = 0;
    cufftComplex *dev_b = 0;
    cufftComplex *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseAddComplexMatrices<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

cudaError_t hostPointwiseSubtractComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny){
	cufftComplex *dev_a = 0;
    cufftComplex *dev_b = 0;
    cufftComplex *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseSubtractComplexMatrices<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
cudaError_t hostPointwiseMultiplyComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny){
	cufftComplex *dev_a = 0;
    cufftComplex *dev_b = 0;
    cufftComplex *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseMultiplyComplexMatrices<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

cudaError_t hostPointwiseDivideComplexMatrices(cufftComplex *c, cufftComplex *a, cufftComplex *b, int Nx, int Ny){
	cufftComplex *dev_a = 0;
    cufftComplex *dev_b = 0;
    cufftComplex *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseDivideComplexMatrices<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
cudaError_t hostPointwiseRealScaleComplexMatrix(cufftComplex *c, cufftComplex *a, float b, int Nx, int Ny){
	cufftComplex *dev_a = 0;
    cufftComplex *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseRealScaleComplexMatrix<<<dimGrid, dimBlock>>>(dev_c, dev_a, b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}
cudaError_t hostPointwiseComplexScaleComplexMatrix(cufftComplex *c, cufftComplex *a, cufftComplex b, int Nx, int Ny){
	cufftComplex *dev_a = 0;
    cufftComplex *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseComplexScaleComplexMatrix<<<dimGrid, dimBlock>>>(dev_c, dev_a, b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}
cudaError_t hostPointwiseAddRealConstantToComplexMatrix(cufftComplex *c, cufftComplex *a, float b, int Nx, int Ny){
	cufftComplex *dev_a = 0;
    cufftComplex *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseAddRealConstantToComplexMatrix<<<dimGrid, dimBlock>>>(dev_c, dev_a, b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}
cudaError_t hostPointwiseAddComplexConstantToComplexMatrix(cufftComplex *c, cufftComplex *a, cufftComplex b, int Nx, int Ny){
	cufftComplex *dev_a = 0;
    cufftComplex *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseAddComplexConstantToComplexMatrix<<<dimGrid, dimBlock>>>(dev_c, dev_a, b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}

cudaError_t hostPointwiseNaturalLogComplexMatrix(cufftComplex *c, cufftComplex *a, int Nx, int Ny){
	cufftComplex *dev_a = 0;
    cufftComplex *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseNaturalLogComplexMatrix<<<dimGrid, dimBlock>>>(dev_c, dev_a, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}

cudaError_t hostPointwisePowerComplexMatrix(cufftComplex *c, cufftComplex *a, int p,int Nx, int Ny){
	cufftComplex *dev_a = 0;
    cufftComplex *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwisePowerComplexMatrix<<<dimGrid, dimBlock>>>(dev_c, dev_a, p, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}

/* Real matrices */
cudaError_t hostPointwiseAddRealMatrices(float *c, float *a, float *b, int Nx, int Ny){
	float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseAddRealMatrices<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

cudaError_t hostPointwiseSubtractRealMatrices(float *c, float *a, float *b, int Nx, int Ny){
	float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseSubtractRealMatrices<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
cudaError_t hostPointwiseMultiplyRealMatrices(float *c, float *a, float *b, int Nx, int Ny){
	float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseMultiplyRealMatrices<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

cudaError_t hostPointwiseDivideRealMatrices(float *c, float *a, float *b, int Nx, int Ny){
	float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseDivideRealMatrices<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
cudaError_t hostPointwiseRealScaleRealMatrix(float *c, float *a, float b, int Nx, int Ny){
	float *dev_a = 0;
    float *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseRealScaleRealMatrix<<<dimGrid, dimBlock>>>(dev_c, dev_a, b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}

cudaError_t hostPointwiseAddRealConstantToRealMatrix(float *c, float *a, float b, int Nx, int Ny){
	float *dev_a = 0;
    float *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseAddRealConstantToRealMatrix<<<dimGrid, dimBlock>>>(dev_c, dev_a, b, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}


cudaError_t hostPointwiseNaturalLogRealMatrix(float *c, float *a, int Nx, int Ny){
	float *dev_a = 0;
    float *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwiseNaturalLogRealMatrix<<<dimGrid, dimBlock>>>(dev_c, dev_a, Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}

cudaError_t hostPointwisePowerRealMatrix(float *c, float *a, int p,int Nx, int Ny){
	float *dev_a = 0;
    float *dev_c = 0;
	int size = Nx * Ny;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	/* Compute the execution configuration NB: block_size_x*block_size_y = number of threads
       On our GPU number of threads < ???
    */
    int block_size_x = BLOCKSIZEX_POINTWISE;
    int block_size_y = BLOCKSIZEY_POINTWISE;
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid (Nx/dimBlock.x, Ny/dimBlock.y); 

    /* Handle N not multiple of block_size_x or block_size_y */
    if (Nx % block_size_x !=0 ) dimGrid.x+=1;
    if (Ny % block_size_y !=0 ) dimGrid.y+=1;

	/* Execute device function */
	pointwisePowerRealMatrix<<<dimGrid, dimBlock>>>(dev_c, dev_a, p,Nx, Ny);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}


#endif
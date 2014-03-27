/* Class to read and write tiff files */

#ifndef TOOLBOX_H
#define TOOLBOX_H
 
#pragma once

#include <stdio.h>
#include <cufft.h>
#include <vector>

#define BLOCKSIZEX_TOOLBOX 16
#define BLOCKSIZEY_TOOLBOX 16

using namespace std;

/* device function declaration */
__global__ void real2complex(float *a, cufftComplex *c, int Nx, int Ny);
__global__ void complex2real_scaled(cufftComplex *c, float *a, int Nx, int Ny,float scale);
__global__ void genIntSequence(float *seq, int start, int end);

/* host function declarations */
char **getFilenames(char* srcFolder, char* prefix, int start, int end);
char *createFullFilename(char* path, char* filename);
void quitProgramPrompt(bool success);
float *linspace(float a, float b, int n);
float* toFloatArray(float** image_rows, int width, int height);
float** toFloat2D(float *image, int width, int height);
void printMatrix1D(char *matrixName, float *matrix, int height, int width);
void printMatrix2D(char *matrixName, float **matrix, int height, int width);
void printComplexMatrix1D(char *matrixName, cufftComplex *matrix, int height, int width);
void printComplexMatrix2D(char *matrixName, cufftComplex **matrix, int height, int width);
cufftComplex *toComplexArray(float *array, int size);
float *toRealArray(cufftComplex *array, int size);
inline int roundUpDiv(int num, int denom){	return (num/denom) + (!(num%denom)? 0 : 1); }
cudaError_t hostGenIntSequence(int start, int end);

/* device functions */

/*Copy real data to complex data */
__global__ void real2complex(float *a, cufftComplex *c, int Nx, int Ny) 
{
	/* compute idx and idy, the location of the element in the original NxN array */
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;

	if ( idx < Nx && idy < Ny) {
		int index = idx + idy*Nx;
		c[index].x = a[index];
		c[index].y = 0.f;
	}
}

/*Copy real part of complex data into real array and apply scaling */
__global__ void complex2real_scaled(cufftComplex *c, float *a, int Nx, int Ny,float scale)
{
    /* compute idx and idy, the location of the element in the original NxN array */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if ( idx < Nx && idy < Ny) {
        int index = idx + idy*Nx;
        a[index] = scale*c[index].x;
    }
} 

__global__ void genIntSequence(float *seq, int start, int end){
	int i = threadIdx.x;
	if(i < (end-start+1)){
		seq[i] = start + i;
	}
    
}

/* host functions */
void quitProgramPrompt(bool success)
{
  int c;
  if(success)
	printf( "\nProgram Executed Successfully. Press ENTER to quit program...\n" );
  else
	printf( "\nProgram Execution Failed. Press ENTER to quit program...\n" );
  fflush( stdout );
  do c = getchar(); while ((c != '\n') && (c != EOF));
}

char **getFilenames(char* srcFolder, char* prefix, int start, int end){

	int numFiles = end - start + 1;
	char **filenames = (char **) malloc(sizeof(char *) * numFiles);

	for(int i = 0;i < numFiles; i++) {
		char *numString = (char *) malloc(sizeof(char) * 5);
		itoa(start+i, numString,10);
		filenames[i] = (char *) malloc(sizeof(char) * (6 + strlen(srcFolder) + strlen(prefix) + strlen(numString))); //6 because null character, .tif, and forward slash
		strcpy(filenames[i], srcFolder);
		strcat(filenames[i], "\\");
		strcat(filenames[i], prefix);
		strcat(filenames[i], numString);
		strcat(filenames[i], ".tif");
	}
	return filenames;
}

char *createFullFilename(char* path, char* filename){
	char *output = (char *) malloc(sizeof(char) * (strlen(path) + strlen(filename) + 2));
	strcpy(output, path);
	strcat(output, "\\");
	strcat(output, filename);
	return output;
}

float* linspace(float a, float b, int n) {
    vector<float> array;
    float *output = (float *) malloc(sizeof(float) * n);
	double step = (b-a) / (n-1);

    while(a <= b) {
        array.push_back(a);
        a += step;           // could recode to better handle rounding errors
    }
	
	for(int i = 0; i < array.size();i++){
		output[i] = array.at(i);
	}

    return output;
}

float* toFloatArray(float** image_rows, int width, int height)
{
	float* output = (float *) calloc(width*height, sizeof(float));
	float* buffer = (float *) calloc(width, sizeof(float));
	for(int i = 0; i < height; i++){
		memcpy(buffer, image_rows[i], width*sizeof(float));
		for(int j = 0; j < width; j++){
			output[j+(i*width)] = buffer[j]; 
		}
	}
	return output;
}

float** toFloat2D(float *image, int width, int height){
	float** image_rows = (float**) calloc(height, sizeof(float *));
	image_rows[0] = image;
	for (int i=1; i<height; i++) {
		image_rows[i] = image_rows[i-1] + width;
	}
	return image_rows;
}

/* Height is the number of rows (x) and Width is the number of columns (y)*/
void printMatrix1D(char *matrixName, float *matrix, int height, int width){
	printf("\n%s=\n", matrixName);
	for(int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			printf("%f\t", matrix[row*width + col]);
		}
		printf("\n");
	}
	printf("\n");
}

void printMatrix2D(char *matrixName, float **matrix, int height, int width){
	printf("\n%s=\n", matrixName);
	for(int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			printf("%f\t", matrix[row][col]);
		}
	}
}

void printComplexMatrix1D(char *matrixName, cufftComplex *matrix, int height, int width){
	printf("\n%s=\n", matrixName);
	for(int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if(matrix[row*width + col].y >= 0){
				printf("%f+%fi\t", matrix[row*width + col].x, matrix[row*width + col].y);
			}else {
				printf("%f%fi\t", matrix[row*width + col].x, matrix[row*width + col].y);
			}
			
		}
		printf("\n");
	}
	printf("\n");
}

void printComplexMatrix2D(char *matrixName, cufftComplex **matrix, int height, int width){
	printf("\n%s=\n", matrixName);
	for(int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if(matrix[row][col].y >= 0){
				printf("%f+%fi\t", matrix[row][col].x, matrix[row][col].y);
			}else {
				printf("%f%fi\t", matrix[row][col].x, matrix[row][col].y);
			}
		}
	}
}

cufftComplex *toComplexArray(float *array, int size){
	cufftComplex *output = (cufftComplex *) malloc(sizeof(cufftComplex)*size);
	for (int i = 0; i < size; i++) {
		output[i].x = array[i];
		output[i].y = 0;
	}
	return output;
}

float *toRealArray(cufftComplex *array, int size){
	float *output = (float *) malloc(sizeof(float)*size);
	for (int i = 0; i < size; i++) {
		output[i] = array[i].x;
	}
	return output;
}

cudaError_t hostGenIntSequence(float *a, int start, int end){
	float *dev_a = 0;
	int size = end-start+1;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for vector
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	/* Execute device function */
	//hostGenIntSequence(float *a, int start, int end)
	genIntSequence<<<1, size>>>(dev_a, start, end);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
	cudaFree(dev_a);

	return cudaStatus;

}



 #endif

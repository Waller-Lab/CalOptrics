//	co_arrays.cpp
//  orginal author: Diivanand Ramalingam
//  original institution: Computational Optical Imaging Lab at UC Berkeley (Prof. Laura Waller's Lab)
//  additional authors: <insert authors here if they're modifying/adding to this file in the library>
//  additional institutions: <insert institutions here if they're modifying/adding to this file in the library>
//	This file is part of the open source project CalOptrics.
//
//	CalOptrics is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  CalOptrics is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with CalOptrics.  If not, see <http://www.gnu.org/licenses/>.

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "co_arrays.h"

namespace co {
	//class DimN member functions
	DimN::DimN(unsigned numDims, unsigned const dimSizes[])
	{
		this->n = numDims;
		this->dimVals = new unsigned[n];
		for(int i = 0; i < n; i++)
			dimVals[i] = dimSizes[i];
	}
	DimN::~DimN()
	{
		delete[] dimVals;
	}
	unsigned DimN::dims() const
	{
		return this->n;
	}
	unsigned DimN::dim(unsigned index)
	{
		return dimVals[index];
	}

	//CudaArray member functions
	
	//Constructors
	template<class T> CudaArray<T>::CudaArray(unsigned size)
	{
		unsigned dimmy[1];
		dimmy[0] = size;
		this->h_vec_ptr = new thrust::host_vector<T>(size);
		this->d_vec_ptr = new thrust::device_vector<T>(size);
		this->dimNptr = new DimN(1, dimmy);
	}
	
	
	template<class T> CudaArray<T>::CudaArray(unsigned size, std::vector<T>* arr){
		unsigned dimmy[1];
		dimmy[0] = size;
		this->h_vec_ptr = new thrust::host_vector<T>(*arr);
		this->d_vec_ptr = new thrust::device_vector<T>(*arr);
		this->dimNptr = new DimN(1, dimmy);
	}

	template<class T> CudaArray<T>::CudaArray(unsigned nrows, unsigned ncols){
		unsigned dimmy[2];
		dimmy[0] = nrows;
		dimmy[1] = ncols;
		this->h_vec_ptr = new thrust::host_vector<T>(nrows*ncols);
		this->d_vec_ptr = new thrust::device_vector<T>(nrows*ncols);
		this->dimNptr = new DimN(2, dimmy);
	}

	template<class T> CudaArray<T>::CudaArray(unsigned nrows, unsigned ncols, std::vector<T>* arr){
		unsigned dimmy[2];
		dimmy[0] = nrows;
		dimmy[1] = ncols;
		this->h_vec_ptr = new thrust::host_vector<T>(*arr);
		this->d_vec_ptr = new thrust::device_vector<T>(*arr);
		this->dimNptr = new DimN(2, dimmy);
	}
	
	//member fucntions
	template<class T> unsigned CudaArray<T>::dims() const
	{
		return this->dimNptr->dims();
	}

	template<class T> unsigned CudaArray<T>::elements() const
	{
		unsigned numElements = 1;
		for(int i = 0; i < this->dimNptr->dims(); i++)
			numElements *= this->dimNptr->dim(i);
		return numElements;
	}

	template<class T> bool CudaArray<T>::isScalar() const
	{
		return this->dimNptr->dims() == 1;
	}

	template<class T> bool CudaArray<T>::isRowVector() const
	{
		return this->dimNptr->dims() == 2 && this->dimNptr->dim(0) == 1; 
	}

	template<class T> bool CudaArray<T>::isColumnVector() const
	{
		return this->dimNptr->dims() == 2 && this->dimNptr->dim(1) == 1; 
	}
	
	
	template<class T> CudaArray<T>::~CudaArray()
	{
		delete h_vec_ptr;
		delete d_vec_ptr;
		delete dimNptr;
	}

	//CudaArray nonmember operator overloads
	/*
	template<class T> CudaArray<T> operator+(CudaArray<T> c1, CudaArray<T> c2)
	{
		CudaArray<T> output = CudaArray<T>()
	}
	template<class T> CudaArray<T> operator-(CudaArray<T> c1, CudaArray<T> c2);
	template<class T> CudaArray<T> operator*(CudaArray<T> c1, CudaArray<T> c2);
	template<class T> CudaArray<T> operator/(CudaArray<T> c1, CudaArray<T> c2);

	template<class T> bool operator==(CudaArray<T> c1, CudaArray<T> c2);
	template<class T> bool operator!=(CudaArray<T> c1, CudaArray<T> c2);
	*/
	//Various CudaArray functions

}
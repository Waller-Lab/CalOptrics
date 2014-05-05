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
#include <thrust/functional.h>
#include <thrust/transform.h>
#include "co_arrays.h"
#include "co_datatypes.h"

namespace co 
{
	//class DimN member functions
	DimN::DimN(std::vector<unsigned>& dimSizes)
	{
		
		dimVals = std::vector<unsigned>();
		this->n = dimSizes.size();
		for(int i = 0; i < dimSizes.size(); i++)
			dimVals.push_back(dimSizes[i]);
	}
	DimN::~DimN()
	{
		// empty the vector
		dimVals.clear();
		std::vector<unsigned>().swap(dimVals);
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
	/*
	template<typename T> CudaArray<T>::CudaArray(unsigned size)
	{
		unsigned dimmy[1];
		dimmy[0] = size;
		this->h_vec_ptr = new thrust::host_vector<T>(size);
		this->d_vec_ptr = new thrust::device_vector<T>(size);
		this->dimNptr = new DimN(1, dimmy);
	}
	*/
	
	
	template<typename T> CudaArray<T>::CudaArray(unsigned size, T init_val){
		std::vector<unsigned> dimmy = std::vector<unsigned>();
		dimmy.push_back(size);
		this->d_vec = thrust::device_vector<T>(size);
		thrust::fill(d_vec.begin(), d_vec.begin() + d_vec.size(), init_val);
		this->dimNptr = new DimN(dimmy);
	}

	/*
	template<typename T> CudaArray<T>::CudaArray(unsigned nrows, unsigned ncols){
		unsigned dimmy[2];
		dimmy[0] = nrows;
		dimmy[1] = ncols;
		this->h_vec_ptr = new thrust::host_vector<T>(nrows*ncols);
		this->d_vec_ptr = new thrust::device_vector<T>(nrows*ncols);
		this->dimNptr = new DimN(2, dimmy);
	}
	*/
	

	template<typename T> CudaArray<T>::CudaArray(unsigned nrows, unsigned ncols, T init_val){
		std::vector<unsigned> dimmy = std::vector<unsigned>();
		dimmy.push_back(nrows);
		dimmy.push_back(ncols);
		this->d_vec = thrust::device_vector<T>(nrows*ncols);
		thrust::fill(d_vec.begin(), d_vec.begin() + d_vec.size(), init_val);
		this->dimNptr = new DimN(dimmy);
	}

	/*
	template<typename T> CudaArray<T>::CudaArray(unsigned nrows, unsigned ncols, std::vector<T>* arr){
		unsigned dimmy[2];
		dimmy[0] = nrows;
		dimmy[1] = ncols;
		this->h_vec_ptr = new thrust::host_vector<T>(*arr);
		this->d_vec_ptr = new thrust::device_vector<T>(*arr);
		this->dimNptr = new DimN(2, dimmy);
	}
	*/

	/*
	template<typename T> CudaArray<T>::CudaArray(DimN dim)
	{
		this->dimNptr = &dim;
	}
	*/
	
	//member fucntions
	template<typename T> unsigned CudaArray<T>::dims() const
	{
		return this->dimNptr->dims();
	}

	template<typename T> unsigned CudaArray<T>::elements() const
	{
		//unsigned numElements = 1;
		//for(int i = 0; i < this->dimNptr->dims(); i++)
		//	numElements *= this->dimNptr->dim(i);
		return d_vec.size();
	}

	template<typename T> bool CudaArray<T>::isScalar() const
	{
		return this->elements() == 1;
	}

	template<typename T> bool CudaArray<T>::isRowVector() const
	{
		return this->dimNptr->dims() == 2 && this->dimNptr->dim(0) == 1; 
	}

	template<typename T> bool CudaArray<T>::isColumnVector() const
	{
		return this->dimNptr->dims() == 2 && this->dimNptr->dim(1) == 1; 
	}
	
	template<typename T> DimN CudaArray<T>::getDimN() const
	{
		return *dimNptr;
	}

	template<typename T> CudaArray<T>::~CudaArray()
	{
		delete dimNptr;
		// empty the vector
		d_vec.clear();
		// deallocate any capacity which may currently be associated with vec
		d_vec.shrink_to_fit();
	}

	//Various CudaArray nonmember functions
	template<typename T> void plus(CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), c1Array.d_vec.begin(), thrust::plus<T>());
	}
	template<typename T> void plus(CudaArray<T>& outArray, CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), outArray.d_vec.begin(), thrust::plus<T>());
	}

	template<typename T> void minus(CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), c1Array.d_vec.begin(), thrust::minus<T>());
	}
	template<typename T> void minus(CudaArray<T>& outArray, CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), outArray.d_vec.begin(), thrust::minus<T>());
	}
	
	template<typename T> void multiplies(CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), c1Array.d_vec.begin(), thrust::multiplies<T>());
	}
	template<typename T> void multiplies(CudaArray<T>& outArray, CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), outArray.d_vec.begin(), thrust::multiplies<T>());
	}

	template<typename T> void divides(CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), c1Array.d_vec.begin(), thrust::divides<T>());
	}
	template<typename T> void divides(CudaArray<T>& outArray, CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), outArray.d_vec.begin(), thrust::divides<T>());
	}

	template<typename T> void negate(CudaArray<T>& c1Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c1Array.d_vec.begin(), thrust::negate<T>());
	}
	template<typename T> void negate(CudaArray<T>& outArray, CudaArray<T>& c1Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), outArray.d_vec.begin(), thrust::negate<T>());
	}

	//not
	template<typename T> void not<T>(CudaArray<T>& c1Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c1Array.d_vec.begin(), thrust::logical_not<T>());
	}
	template<typename T> void not<T>(CudaArray<T>& outArray, CudaArray<T>& c1Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), outArray.d_vec.begin(), thrust::logical_not<T>());
	}

	//and
	template<typename T> void and(CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), c1Array.d_vec.begin(), thrust::logical_and<T>());
	}
	template<typename T> void and(CudaArray<T>& outArray, CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), outArray.d_vec.begin(), thrust::logical_and<T>());
	}

	//or
	template<typename T> void or(CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), c1Array.d_vec.begin(), thrust::logical_or<T>());
	}
	template<typename T> void or(CudaArray<T>& outArray, CudaArray<T>& c1Array, CudaArray<T>& c2Array)
	{
		thrust::transform(c1Array.d_vec.begin(), c1Array.d_vec.end(), c2Array.d_vec.begin(), outArray.d_vec.begin(), thrust::logical_or<T>());
	}

	template<typename T> void print_matrix(std::string name, CudaArray<T> A)
	{
		int nr_rows_A = A.getDimN().dim(0);
		int nr_cols_A = A.getDimN().dim(1);

		//std::cout << nr_rows_A << std::endl;
		//std::cout << nr_cols_A << std::endl;

		std::cout << name << "=" << std::endl;
		//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
		for(int i = 0; i < nr_rows_A; ++i){
			for(int j = 0; j < nr_cols_A; ++j){
				T item = A.d_vec[j * nr_rows_A + i];
				std::cout << item << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	
	}

	template<typename T> void print_array(std::string name, CudaArray<T> A)
	{
		std::cout << name << "=" << std::endl;
		for(int i = 0;i < A.elements(); i++){
			T item = A.d_vec[i];
			std::cout << item << std::endl;
		}
		std::cout << std::endl;
	}

	//create your own unary or binary function example:
	/*
	 struct sine : public thrust::unary_function<float,float>
	{
		__host__ __device__
		float operator()(float x) { return sinf(x); }
	};

	struct exponentiate : public thrust::binary_function<float,float,float>
	{
		__host__ __device__
		float operator()(float x, float y) { return powf(x,y); }
	};
	*/

	//The explicit instantiation part

	//explicit instantiation of types
	template class CudaArray<bool>; 
	template class CudaArray<int>;
	template class CudaArray<unsigned>;
	template class CudaArray<float>;
	template class CudaArray<cfloat>; 
	template class CudaArray<double>;
	template class CudaArray<cdouble>;

	//explicity instantitation of function templates

	//plus
	template void plus<int>(CudaArray<int>& c1Array, CudaArray<int>& c2Array);
	template void plus<int>(CudaArray<int>& outArray, CudaArray<int>& c1Array, CudaArray<int>& c2Array);
	template void plus<unsigned>(CudaArray<unsigned>& c1Array, CudaArray<unsigned>& c2Array);
	template void plus<unsigned>(CudaArray<unsigned>& outArray, CudaArray<unsigned>& c1Array, CudaArray<unsigned>& c2Array);
	template void plus<float>(CudaArray<float>& c1Array, CudaArray<float>& c2Array);
	template void plus<float>(CudaArray<float>& outArray, CudaArray<float>& c1Array, CudaArray<float>& c2Array);
	template void plus<double>(CudaArray<double>& c1Array, CudaArray<double>& c2Array);
	template void plus<double>(CudaArray<double>& outArray, CudaArray<double>& c1Array, CudaArray<double>& c2Array);
	template void plus<cfloat>(CudaArray<cfloat>& c1Array, CudaArray<cfloat>& c2Array);
	template void plus<cfloat>(CudaArray<cfloat>& outArray, CudaArray<cfloat>& c1Array, CudaArray<cfloat>& c2Array);
	template void plus<cdouble>(CudaArray<cdouble>& c1Array, CudaArray<cdouble>& c2Array);
	template void plus<cdouble>(CudaArray<cdouble>& outArray, CudaArray<cdouble>& c1Array, CudaArray<cdouble>& c2Array);

	//minus
	template void minus<int>(CudaArray<int>& c1Array, CudaArray<int>& c2Array);
	template void minus<int>(CudaArray<int>& outArray, CudaArray<int>& c1Array, CudaArray<int>& c2Array);
	template void minus<unsigned>(CudaArray<unsigned>& c1Array, CudaArray<unsigned>& c2Array);
	template void minus<unsigned>(CudaArray<unsigned>& outArray, CudaArray<unsigned>& c1Array, CudaArray<unsigned>& c2Array);
	template void minus<float>(CudaArray<float>& c1Array, CudaArray<float>& c2Array);
	template void minus<float>(CudaArray<float>& outArray, CudaArray<float>& c1Array, CudaArray<float>& c2Array);
	template void minus<double>(CudaArray<double>& c1Array, CudaArray<double>& c2Array);
	template void minus<double>(CudaArray<double>& outArray, CudaArray<double>& c1Array, CudaArray<double>& c2Array);
	template void minus<cfloat>(CudaArray<cfloat>& c1Array, CudaArray<cfloat>& c2Array);
	template void minus<cfloat>(CudaArray<cfloat>& outArray, CudaArray<cfloat>& c1Array, CudaArray<cfloat>& c2Array);
	template void minus<cdouble>(CudaArray<cdouble>& c1Array, CudaArray<cdouble>& c2Array);
	template void minus<cdouble>(CudaArray<cdouble>& outArray, CudaArray<cdouble>& c1Array, CudaArray<cdouble>& c2Array);

	//multiplies
	template void multiplies<int>(CudaArray<int>& c1Array, CudaArray<int>& c2Array);
	template void multiplies<int>(CudaArray<int>& outArray, CudaArray<int>& c1Array, CudaArray<int>& c2Array);
	template void multiplies<unsigned>(CudaArray<unsigned>& c1Array, CudaArray<unsigned>& c2Array);
	template void multiplies<unsigned>(CudaArray<unsigned>& outArray, CudaArray<unsigned>& c1Array, CudaArray<unsigned>& c2Array);
	template void multiplies<float>(CudaArray<float>& c1Array, CudaArray<float>& c2Array);
	template void multiplies<float>(CudaArray<float>& outArray, CudaArray<float>& c1Array, CudaArray<float>& c2Array);
	template void multiplies<double>(CudaArray<double>& c1Array, CudaArray<double>& c2Array);
	template void multiplies<double>(CudaArray<double>& outArray, CudaArray<double>& c1Array, CudaArray<double>& c2Array);
	template void multiplies<cfloat>(CudaArray<cfloat>& c1Array, CudaArray<cfloat>& c2Array);
	template void multiplies<cfloat>(CudaArray<cfloat>& outArray, CudaArray<cfloat>& c1Array, CudaArray<cfloat>& c2Array);
	template void multiplies<cdouble>(CudaArray<cdouble>& c1Array, CudaArray<cdouble>& c2Array);
	template void multiplies<cdouble>(CudaArray<cdouble>& outArray, CudaArray<cdouble>& c1Array, CudaArray<cdouble>& c2Array);


	//divides
	template void divides<int>(CudaArray<int>& c1Array, CudaArray<int>& c2Array);
	template void divides<int>(CudaArray<int>& outArray, CudaArray<int>& c1Array, CudaArray<int>& c2Array);
	template void divides<unsigned>(CudaArray<unsigned>& c1Array, CudaArray<unsigned>& c2Array);
	template void divides<unsigned>(CudaArray<unsigned>& outArray, CudaArray<unsigned>& c1Array, CudaArray<unsigned>& c2Array);
	template void divides<float>(CudaArray<float>& c1Array, CudaArray<float>& c2Array);
	template void divides<float>(CudaArray<float>& outArray, CudaArray<float>& c1Array, CudaArray<float>& c2Array);
	template void divides<double>(CudaArray<double>& c1Array, CudaArray<double>& c2Array);
	template void divides<double>(CudaArray<double>& outArray, CudaArray<double>& c1Array, CudaArray<double>& c2Array);
	template void divides<cfloat>(CudaArray<cfloat>& c1Array, CudaArray<cfloat>& c2Array);
	template void divides<cfloat>(CudaArray<cfloat>& outArray, CudaArray<cfloat>& c1Array, CudaArray<cfloat>& c2Array);
	template void divides<cdouble>(CudaArray<cdouble>& c1Array, CudaArray<cdouble>& c2Array);
	template void divides<cdouble>(CudaArray<cdouble>& outArray, CudaArray<cdouble>& c1Array, CudaArray<cdouble>& c2Array);

	//negate
	template void negate<int>(CudaArray<int>& c1Array);
	template void negate<int>(CudaArray<int>& outArray, CudaArray<int>& c1Array);
	template void negate<unsigned>(CudaArray<unsigned>& c1Array);
	template void negate<unsigned>(CudaArray<unsigned>& outArray, CudaArray<unsigned>& c1Array);
	template void negate<float>(CudaArray<float>& c1Array);
	template void negate<float>(CudaArray<float>& outArray, CudaArray<float>& c1Array);
	template void negate<double>(CudaArray<double>& c1Array);
	template void negate<double>(CudaArray<double>& outArray, CudaArray<double>& c1Array);
	template void negate<cfloat>(CudaArray<cfloat>& c1Array);
	template void negate<cfloat>(CudaArray<cfloat>& outArray, CudaArray<cfloat>& c1Array);
	template void negate<cdouble>(CudaArray<cdouble>& c1Array);
	template void negate<cdouble>(CudaArray<cdouble>& outArray, CudaArray<cdouble>& c1Array);


	//not
	template void not<bool>(CudaArray<bool>& c1Array);
	template void not<bool>(CudaArray<bool>& outArray, CudaArray<bool>& c1Array);

	//and
	template void and<bool>(CudaArray<bool>& c1Array, CudaArray<bool>& c2Array);
	template void and<bool>(CudaArray<bool>& outArray, CudaArray<bool>& c1Array, CudaArray<bool>& c2Array);

	//or
	template void or<bool>(CudaArray<bool>& c1Array, CudaArray<bool>& c2Array);
	template void or<bool>(CudaArray<bool>& outArray, CudaArray<bool>& c1Array, CudaArray<bool>& c2Array);

	//print
	template void print_matrix<bool>(std::string name, CudaArray<bool> A);
	template void print_array<bool>(std::string name, CudaArray<bool> A);
	template void print_matrix<int>(std::string name, CudaArray<int> A);
	template void print_array<int>(std::string name, CudaArray<int> A);
	template void print_matrix<unsigned>(std::string name, CudaArray<unsigned> A);
	template void print_array<unsigned>(std::string name, CudaArray<unsigned> A);
	template void print_matrix<float>(std::string name, CudaArray<float> A);
	template void print_array<float>(std::string name, CudaArray<float> A);
	template void print_matrix<double>(std::string name, CudaArray<double> A);
	template void print_array<double>(std::string name, CudaArray<double> A);
	template void print_matrix<cfloat>(std::string name, CudaArray<cfloat> A);
	template void print_array<cfloat>(std::string name, CudaArray<cfloat> A);
	template void print_matrix<cdouble>(std::string name, CudaArray<cdouble> A);
	template void print_array<cdouble>(std::string name, CudaArray<cdouble> A);
}
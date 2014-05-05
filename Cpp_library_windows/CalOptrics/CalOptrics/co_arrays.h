//	co_arrays.h
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

#ifndef CO_ARRAYS_H
#define CO_ARRAYS_H

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace co 
{

	class DimN {
	public:
		DimN(std::vector<unsigned>& dimSizes);
		unsigned dims() const;
		unsigned dim(unsigned index);
		~DimN();
	private:
		unsigned n;
		std::vector<unsigned> dimVals;
	};
	
	/*
	template<class T> class Array {
	public:
		virtual unsigned dims() const = 0;
		virtual unsigned elements() const = 0;
		virtual bool isScalar() const = 0;
		virtual bool isRowVector() const = 0;
		virtual bool isColumnVector() const = 0;
		//virtual vector<T> device() const = 0;
		//virtual vector<T> host() const = 0;
		virtual ~Array() = 0;
	};
	template<class T> Array<T>::~Array() {}
	*/

	template<typename T> class CudaArray {
	public:
		CudaArray(unsigned size, T init_val);
		CudaArray(unsigned nrows, unsigned ncols, T init_val);
		//CudaArray(DimN dim);
		unsigned dims() const;
		unsigned elements() const;
		bool isScalar() const;
		bool isRowVector() const;
		bool isColumnVector() const;
		DimN getDimN() const;
		thrust::device_vector<T> d_vec;
		~CudaArray();
	private:
		DimN *dimNptr;
	};

	//CudaArray nonmember operator overloads
	template<typename T> void plus(CudaArray<T>& c1, CudaArray<T>& c2);
	template<typename T> void plus(CudaArray<T>& out, CudaArray<T>& c1, CudaArray<T>& c2);
	template<typename T> void minus(CudaArray<T>& c1, CudaArray<T>& c2);
	template<typename T> void minus(CudaArray<T>& out, CudaArray<T>& c1, CudaArray<T>& c2);
	template<typename T> void multiplies(CudaArray<T>& c1, CudaArray<T>& c2);
	template<typename T> void multiplies(CudaArray<T>& out, CudaArray<T>& c1, CudaArray<T>& c2);
	template<typename T> void divides(CudaArray<T>& c1, CudaArray<T>& c2);
	template<typename T> void divides(CudaArray<T>& out, CudaArray<T>& c1, CudaArray<T>& c2);
	template<typename T> void and(CudaArray<T>& c1, CudaArray<T>& c2);
	template<typename T> void or(CudaArray<T>& out, CudaArray<T>& c1, CudaArray<T>& c2);
	template<typename T> void not(CudaArray<T>& c1);
	template<typename T> void not(CudaArray<T>& out, CudaArray<T>& c1);
	template<typename T> void negate(CudaArray<T>& c1);
	template<typename T> void negate(CudaArray<T>& out, CudaArray<T>& c1);
	template<typename T> void print_matrix(std::string name, CudaArray<T> A);
	template<typename T> void print_array(std::string name, CudaArray<T> A);
	//template<class T> std::ostream& operator<<(std::ostream& os, CudaArray<T>& c2);
	/*
	template<class T> CudaArray<T> operator+(CudaArray<T> c1, CudaArray<T> c2);
	template<class T> CudaArray<T> operator-(CudaArray<T> c1, CudaArray<T> c2);
	template<class T> CudaArray<T> operator*(CudaArray<T> c1, CudaArray<T> c2);
	template<class T> CudaArray<T> operator/(CudaArray<T> c1, CudaArray<T> c2);

	template<class T> bool operator==(CudaArray<T> c1, CudaArray<T> c2);
	template<class T> bool operator!=(CudaArray<T> c1, CudaArray<T> c2);
	*/
	//Various CudaArray functions

}

#endif
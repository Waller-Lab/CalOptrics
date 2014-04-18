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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace co {
	typedef unsigned int uint;

	class DimN {
	public:
		DimN(uint numDims, uint const *dimSizes);
		uint dims();
		uint dim(uint index);
	private:
		uint n;
		uint *dimVals;
	};

	template<class T> class Array {
	public:
		virtual unsigned dims() const = 0;
		virtual unsigned elements() const = 0;
		virtual bool isScalar() const = 0;
		virtual bool isRowVector() const = 0;
		virtual bool isColumnVector() const = 0;
		virtual T* device() const = 0;
		virtual T* host() const = 0;
		virtual ~COArray() = 0;
	};
	
	template<class T> class CudaArray : public Array {
	public:
		CudaArray(uint size);
		CudaArray(uint size, T* arr);
		CudaArray(uint nrows, uint ncols);
		CudaArray(uint nrows, uint ncols, T* arr);
		unsigned dims() const;
		unsigned elements() const;
		bool isScalar() const;
		bool isRowVector() const;
		bool isColumnVector() const;
		T* device() const;
		T* host() const;
		~CudaArray();
	private:
		DimN dimN;
		thrust::host_vector<T> h_vec;
		thrust::device_vector<T> d_vec;
	};
}

#endif
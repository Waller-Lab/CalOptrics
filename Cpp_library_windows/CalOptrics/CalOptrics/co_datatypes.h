//	co_datatypes.h
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

#ifndef CO_DATATYPES_H
#define CO_DATATYPES_H

#include <cufft.h> 

namespace co {
	template<class T> class CoBool {
	public:
		CoBool(T val);
		bool val();
	private:
		T value;
	};

	template<class T> class CoFloat {
	public:
		CoFloat(T val);
		float val();
	private:
		T value;
	};

	template<class T> class CoCFloat {
	public:
		CoCFloat(T val);
		cufftComplex val();
	private:
		T value;
	};

	template<class T> class CoDouble {
	public:
		CoDouble(T val);
		double val();
	private:
		T value;
	};

	template<class T> class CoCDouble {
	public:
		CoCDouble(T val);
		cufftDoubleComplex val();
	private:
		T value;
	};

	template<class T> class CoSInt {
	public:
		int val();
	};

	template<class T> class CoUInt {
	public:
		unsigned val();
	};

	typedef CoBool<bool> Bool;
	typedef CoFloat<float> Float;
	typedef CoCFloat<cufftComplex> CFloat;
	typedef CoDouble<double> Double;
	typedef CoCDouble<cufftDoubleComplex> CDouble;
	typedef CoSInt<int> Int;
	typedef CoUInt<unsigned> UInt;
}

#endif
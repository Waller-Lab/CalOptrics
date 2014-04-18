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
	template<> class CoBool<bool> {
	public:
		CoBool(bool val);
		bool val();
	private:
		bool value;
	};

	template<> class CoFloat<float> {
	public:
		CoFloat(float val);
		float val();
	private:
		float value;
	};

	template<> class CoCFloat<cufftComplex> {
	public:
		CoCFloat(cufftComplex val);
		CoCFloat(float re, float im);
		cufftComplex val() const;
		// member operator overloads
		CoCFloat<cufftComplex>& operator+=(CoCFloat<cufftComplex> c)
		CoCFloat<cufftComplex>& operator+=(float c)
		CoCFloat<cufftComplex>& operator-=(CoCFloat<cufftComplex> c)
		CoCFloat<cufftComplex>& operator-=(float c)
		CoCFloat<cufftComplex>& operator*=(CoCFloat<cufftComplex> c)
		CoCFloat<cufftComplex>& operator*=(float c)
		CoCFloat<cufftComplex>& operator/=(CoCFloat<cufftComplex> c)
		CoCFloat<cufftComplex>& operator/=(float c)
	private:
		cufftComplex value;
	};

	//start CoCFloat function declarations
	
	CoCFloat<cufftComplex> operator+(CoCFloat<cufftComplex> c1, CoCFloat<cufftComplex> c2);
	CoCFloat<cufftComplex> operator+(CoCFloat<cufftComplex> c1, float d);
	CoCFloat<cufftComplex> operator+(float d, CoCFloat<cufftComplex> c1);

	CoCFloat<cufftComplex> operator-(CoCFloat<cufftComplex> c1, CoCFloat<cufftComplex> c2);
	CoCFloat<cufftComplex> operator-(CoCFloat<cufftComplex> c1, float d);
	CoCFloat<cufftComplex> operator-(float d, CoCFloat<cufftComplex> c1);

	CoCFloat<cufftComplex> operator*(CoCFloat<cufftComplex> c1, CoCFloat<cufftComplex> c2);
	CoCFloat<cufftComplex> operator*(CoCFloat<cufftComplex> c1, float d);
	CoCFloat<cufftComplex> operator*(float d, CoCFloat<cufftComplex> c1);

	CoCFloat<cufftComplex> operator/(CoCFloat<cufftComplex> c1, CoCFloat<cufftComplex> c2);
	CoCFloat<cufftComplex> operator/(CoCFloat<cufftComplex> c1, float d);
	CoCFloat<cufftComplex> operator/(float d, CoCFloat<cufftComplex> c1);

	CoCFloat<cufftComplex> operator-(CoCFloat<cufftComplex> c1); //unary minus
	CoCFloat<cufftComplex> operator+(CoCFloat<cufftComplex> c1); //unary plus

	bool operator==(CoCFloat<cufftComplex> c1, CoCFloat<cufftComplex> c2);
	bool operator!=(CoCFloat<cufftComplex> c1, CoCFloat<cufftComplex> c2);

	istream& operator>>(istream&, CoCFloat<cufftComplex>& c1); //input
	istream& operator<<(istream&, CoCFloat<cufftComplex>& c1); //output

	CoCFloat<cufftComplex> polar(double rho, double theta);
	CoCFloat<cufftComplex> conj(CoCFloat<cufftComplex> c);

	CoCFloat<cufftComplex> abs(CoCFloat<cufftComplex> c);
	CoCFloat<cufftComplex> arg(CoCFloat<cufftComplex> c);
	CoCFloat<cufftComplex> norm(CoCFloat<cufftComplex> c);

	CoCFloat<cufftComplex> real(CoCFloat<cufftComplex> c);
	CoCFloat<cufftComplex> imag(CoCFloat<cufftComplex> c);

	//end CoCFloat function declarations

	template<> class CoDouble<double> {
	public:
		CoDouble(double val);
		double val() const;
	private:
		double value;
	};

	template<> class CoCDouble<cufftDoubleComplex> {
	public:
		CoCDouble(cufftDoubleComplex val);
		CoCDouble(double re, double im);
		cufftDoubleComplex val() const;
		// member operator overloads
		CoCDouble<cufftDoubleComplex>& operator+=(CoCDouble<cufftDoubleComplex> c)
		CoCDouble<cufftDoubleComplex>& operator+=(double c)
		CoCDouble<cufftDoubleComplex>& operator-=(CoCDouble<cufftDoubleComplex> c)
		CoCDouble<cufftDoubleComplex>& operator-=(double c)
		CoCDouble<cufftDoubleComplex>& operator*=(CoCDouble<cufftDoubleComplex> c)
		CoCDouble<cufftDoubleComplex>& operator*=(double c)
		CoCDouble<cufftDoubleComplex>& operator/=(CoCDouble<cufftDoubleComplex> c)
		CoCDouble<cufftDoubleComplex>& operator/=(double c)
	private:
		cufftComplexComplex value;
	};

	//start CoCDouble function declarations
	
	CoCDouble<cufftDoubleComplex> operator+(CoCDouble<cufftDoubleComplex> c1, CoCDouble<cufftDoubleComplex> c2);
	CoCDouble<cufftDoubleComplex> operator+(CoCDouble<cufftDoubleComplex> c1, double d);
	CoCDouble<cufftDoubleComplex> operator+(double d, CoCDouble<cufftDoubleComplex> c1);

	CoCDouble<cufftDoubleComplex> operator-(CoCDouble<cufftDoubleComplex> c1, CoCDouble<cufftDoubleComplex> c2);
	CoCDouble<cufftDoubleComplex> operator-(CoCDouble<cufftDoubleComplex> c1, double d);
	CoCDouble<cufftDoubleComplex> operator-(double d, CoCDouble<cufftDoubleComplex> c1);

	CoCDouble<cufftDoubleComplex> operator*(CoCDouble<cufftDoubleComplex> c1, CoCDouble<cufftDoubleComplex> c2);
	CoCDouble<cufftDoubleComplex> operator*(CoCDouble<cufftDoubleComplex> c1, double d);
	CoCDouble<cufftDoubleComplex> operator*(double d, CoCDouble<cufftDoubleComplex> c1);

	CoCDouble<cufftDoubleComplex> operator/(CoCDouble<cufftDoubleComplex> c1, CoCDouble<cufftDoubleComplex> c2);
	CoCDouble<cufftDoubleComplex> operator/(CoCDouble<cufftDoubleComplex> c1, double d);
	CoCDouble<cufftDoubleComplex> operator/(double d, CoCDouble<cufftDoubleComplex> c1);

	CoCDouble<cufftDoubleComplex> operator-(CoCDouble<cufftDoubleComplex> c1); //unary minus
	CoCDouble<cufftDoubleComplex> operator+(CoCDouble<cufftDoubleComplex> c1); //unary plus

	bool operator==(CoCDouble<cufftDoubleComplex> c1, CoCDouble<cufftDoubleComplex> c2);
	bool operator!=(CoCDouble<cufftDoubleComplex> c1, CoCDouble<cufftDoubleComplex> c2);

	istream& operator>>(istream&, CoCDouble<cufftDoubleComplex>& c1); //input
	istream& operator<<(istream&, CoCDouble<cufftDoubleComplex>& c1); //output

	CoCDouble<cufftDoubleComplex> polar(double rho, double theta);
	CoCDouble<cufftDoubleComplex> conj(CoCDouble<cufftDoubleComplex> c);

	CoCDouble<cufftDoubleComplex> abs(CoCDouble<cufftDoubleComplex> c);
	CoCDouble<cufftDoubleComplex> arg(CoCDouble<cufftDoubleComplex> c);
	CoCDouble<cufftDoubleComplex> norm(CoCDouble<cufftDoubleComplex> c);

	CoCDouble<cufftDoubleComplex> real(CoCDouble<cufftDoubleComplex> c);
	CoCDouble<cufftDoubleComplex> imag(CoCDouble<cufftDoubleComplex> c);

	//end CoCDouble function declarations

	template<> class CoSInt<int> {
	public:
		CoSInt(int val);
		int val() const;
	private:
		int value;
	};

	template<> class CoUInt<unsigned> {
	public:
		CoUInt(unsigned val);
		unsigned val() const;
	private:
		unsigned value;
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
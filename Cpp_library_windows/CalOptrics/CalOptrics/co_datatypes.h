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
		bool val() const;
		// member operator overloads
		CoBool<bool>& operator+=(CoBool<bool> c);
		CoBool<bool>& operator+=(bool c);
		CoBool<bool>& operator-=(CoBool<bool> c);
		CoBool<bool>& operator-=(bool c);
		CoBool<bool>& operator*=(CoBool<bool> c);
		CoBool<bool>& operator*=(bool c);
		CoBool<bool>& operator/=(CoBool<bool> c);
		CoBool<bool>& operator/=(bool c);
	private:
		bool value;
	};

	//start CoBool function declarations
	
	CoBool<bool> operator+(CoBool<bool> c1, CoBool<bool> c2);
	CoBool<bool> operator+(CoBool<bool> c1, bool d);
	CoBool<bool> operator+(bool d, CoBool<bool> c1);

	CoBool<bool> operator-(CoBool<bool> c1, CoBool<bool> c2);
	CoBool<bool> operator-(CoBool<bool> c1, bool d);
	CoBool<bool> operator-(bool d, CoBool<bool> c1);

	CoBool<bool> operator*(CoBool<bool> c1,CoBool<bool>c2);
	CoBool<bool> operator*(CoBool<bool> c1, bool d);
	CoBool<bool> operator*(bool d, CoBool<bool> c1);

	CoBool<bool> operator/(CoBool<bool> c1, CoBool<bool> c2);
	CoBool<bool> operator/(CoBool<bool> c1, bool d);
	CoBool<bool> operator/(bool d, CoBool<bool> c1);

	CoBool<bool> operator-(CoBool<bool> c1); //unary minus
	CoBool<bool> operator+(CoBool<bool> c1); //unary plus

	bool operator==(CoBool<bool> c1, CoBool<bool> c2);
	bool operator!=(CoBool<bool> c1, CoBool<bool> c2);

	//istream& operator>>(istream&, CoBool<bool>& c1); //input
	//ostream& operator<<(ostream&, CoBool<bool>& c1); //output

	//end CoBool function declarations

	template<> class CoFloat<float> 
	{
	public:
		CoFloat(float val);
		float val() const;
		// member operator overloads
		CoFloat<float>& operator+=(CoFloat<float> c);
		CoFloat<float>& operator+=(float c);
		CoFloat<float>& operator-=(CoFloat<float> c);
		CoFloat<float>& operator-=(float c);
		CoFloat<float>& operator*=(CoFloat<float> c);
		CoFloat<float>& operator*=(float c);
		CoFloat<float>& operator/=(CoFloat<float> c);
		CoFloat<float>& operator/=(float c);
	private:
		float value;
	};

	//start CoFloat function declarations
	
	CoFloat<float> operator+(CoFloat<float> c1, CoFloat<float> c2);
	CoFloat<float> operator+(CoFloat<float> c1, float d);
	CoFloat<float> operator+(float d, CoFloat<float> c1);

	CoFloat<float> operator-(CoFloat<float> c1, CoFloat<float> c2);
	CoFloat<float> operator-(CoFloat<float> c1, float d);
	CoFloat<float> operator-(float d, CoFloat<float> c1);

	CoFloat<float> operator*(CoFloat<float> c1, CoFloat<float> c2);
	CoFloat<float> operator*(CoFloat<float> c1, float d);
	CoFloat<float> operator*(float d, CoFloat<float> c1);

	CoFloat<float> operator/(CoFloat<float> c1, CoFloat<float> c2);
	CoFloat<float> operator/(CoFloat<float> c1, float d);
	CoFloat<float> operator/(float d, CoFloat<float> c1);

	bool operator==(CoFloat<float> c1, CoFloat<float> c2);
	bool operator!=(CoFloat<float> c1, CoFloat<float> c2);

	//istream& operator>>(istream&, CoFloat<float>& c1); //input
	//ostream& operator<<(ostream&, CoFloat<float>& c1); //output

	//end CoFloat function declarations

	template<> class CoCFloat<cufftComplex> {
	public:
		CoCFloat(cufftComplex val);
		CoCFloat(CoCFloat<cufftComplex> c);
		cufftComplex val() const;
		// member operator overloads
		CoCFloat<cufftComplex>& operator+=(CoCFloat<cufftComplex> c);
		CoCFloat<cufftComplex>& operator+=(cufftComplex c);
		CoCFloat<cufftComplex>& operator+=(float c);
		CoCFloat<cufftComplex>& operator-=(CoCFloat<cufftComplex> c);
		CoCFloat<cufftComplex>& operator-=(cufftComplex c);
		CoCFloat<cufftComplex>& operator-=(float c);
		CoCFloat<cufftComplex>& operator*=(CoCFloat<cufftComplex> c);
		CoCFloat<cufftComplex>& operator*=(cufftComplex c);
		CoCFloat<cufftComplex>& operator*=(float c);
		CoCFloat<cufftComplex>& operator/=(CoCFloat<cufftComplex> c);
		CoCFloat<cufftComplex>& operator/=(cufftComplex c);
		CoCFloat<cufftComplex>& operator/=(float c);
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

	bool operator==(CoCFloat<cufftComplex> c1, CoCFloat<cufftComplex> c2);
	bool operator!=(CoCFloat<cufftComplex> c1, CoCFloat<cufftComplex> c2);

	//istream& operator>>(istream&, CoCFloat<cufftComplex>& c1); //input
	//ostream& operator<<(ostream&, CoCFloat<cufftComplex>& c1); //output

	CoCFloat<cufftComplex> polar(float rho, float theta);
	CoCFloat<cufftComplex> conj(CoCFloat<cufftComplex> c);

	float abs(CoCFloat<cufftComplex> c);
	float arg(CoCFloat<cufftComplex> c);
	float norm(CoCFloat<cufftComplex> c);

	float real(CoCFloat<cufftComplex> c);
	float imag(CoCFloat<cufftComplex> c);

	//end CoCFloat function declarations

	template<> class CoDouble<double> {
	public:
		CoDouble(double val);
		double val() const;
		CoDouble<double>& operator+=(CoDouble<double> c);
		CoDouble<double>& operator+=(double c);
		CoDouble<double>& operator-=(CoDouble<double> c);
		CoDouble<double>& operator-=(double c);
		CoDouble<double>& operator*=(CoDouble<double> c);
		CoDouble<double>& operator*=(double c);
		CoDouble<double>& operator/=(CoDouble<double> c);
		CoDouble<double>& operator/=(double c);
	private:
		double value;
	};

	//start CoDouble function declarations
	
	CoDouble<double> operator+(CoDouble<double> c1, CoDouble<double> c2);
	CoDouble<double> operator+(CoDouble<double> c1, double d);
	CoDouble<double> operator+(double d, CoDouble<double> c1);

	CoDouble<double> operator-(CoDouble<double> c1, CoDouble<double> c2);
	CoDouble<double> operator-(CoDouble<double> c1, double d);
	CoDouble<double> operator-(double d, CoDouble<double> c1);

	CoDouble<double> operator*(CoDouble<double> c1, CoDouble<double> c2);
	CoDouble<double> operator*(CoDouble<double> c1, double d);
	CoDouble<double> operator*(double d, CoDouble<double> c1);

	CoDouble<double> operator/(CoDouble<double> c1, CoDouble<double> c2);
	CoDouble<double> operator/(CoDouble<double> c1, double d);
	CoDouble<double> operator/(double d, CoDouble<double> c1);

	bool operator==(CoDouble<double> c1, CoDouble<double> c2);
	bool operator!=(CoDouble<double> c1, CoDouble<double> c2);

	//istream& operator>>(istream&, CoDouble<double>& c1); //input
	//ostream& operator<<(ostream&, CoDouble<double>& c1); //output

	//end CoDouble function declarations

	template<> class CoCDouble<cufftDoubleComplex> {
	public:
		CoCDouble(cufftDoubleComplex val);
		CoCDouble(CoCFloat<cufftDoubleComplex> c);
		cufftDoubleComplex val() const;
		// member operator overloads
		CoCDouble<cufftDoubleComplex>& operator+=(CoCDouble<cufftDoubleComplex> c);
		CoCDouble<cufftDoubleComplex>& operator+=(cufftDoubleComplex c);
		CoCDouble<cufftDoubleComplex>& operator+=(double c);
		CoCDouble<cufftDoubleComplex>& operator-=(CoCDouble<cufftDoubleComplex> c);
		CoCDouble<cufftDoubleComplex>& operator-=(cufftDoubleComplex c);
		CoCDouble<cufftDoubleComplex>& operator-=(double c);
		CoCDouble<cufftDoubleComplex>& operator*=(CoCDouble<cufftDoubleComplex> c);
		CoCDouble<cufftDoubleComplex>& operator*=(cufftDoubleComplex c);
		CoCDouble<cufftDoubleComplex>& operator*=(double c);
		CoCDouble<cufftDoubleComplex>& operator/=(CoCDouble<cufftDoubleComplex> c);
		CoCDouble<cufftDoubleComplex>& operator/=(cufftDoubleComplex c);
		CoCDouble<cufftDoubleComplex>& operator/=(double c);
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

	bool operator==(CoCDouble<cufftDoubleComplex> c1, CoCDouble<cufftDoubleComplex> c2);
	bool operator!=(CoCDouble<cufftDoubleComplex> c1, CoCDouble<cufftDoubleComplex> c2);

	//istream& operator>>(istream&, CoCDouble<cufftDoubleComplex>& c1); //input
	//ostream& operator<<(ostream&, CoCDouble<cufftDoubleComplex>& c1); //output

	CoCDouble<cufftDoubleComplex> polar(double rho, double theta);
	CoCDouble<cufftDoubleComplex> conj(CoCDouble<cufftDoubleComplex> c);

	double abs(CoCDouble<cufftDoubleComplex> c);
	double arg(CoCDouble<cufftDoubleComplex> c);
	double norm(CoCDouble<cufftDoubleComplex> c);

	double real(CoCDouble<cufftDoubleComplex> c);
	double imag(CoCDouble<cufftDoubleComplex> c);

	//end CoCDouble function declarations

	template<> class CoSInt<int> {
	public:
		CoSInt(int val);
		int val() const;
		// member operator overloads
		CoSInt<int>& operator+=(CoSInt<int> c)
		CoSInt<int>& operator+=(int c)
		CoSInt<int>& operator-=(CoSInt<int> c)
		CoSInt<int>& operator-=(int c)
		CoSInt<int>& operator*=(CoSInt<int> c)
		CoSInt<int>& operator*=(int c)
		CoSInt<int>& operator/=(CoSInt<int> c)
		CoSInt<int>& operator/=(int c)
	private:
		int value;
	};

	//start CoSInt function declarations
	
	CoSInt<int> operator+(CoSInt<int> c1, CoSInt<int> c2);
	CoSInt<int> operator+(CoSInt<int> c1, int d);
	CoSInt<int> operator+(int d, CoSInt<int> c1);

	CoSInt<int> operator-(CoSInt<int> c1, CoSInt<int> c2);
	CoSInt<int> operator-(CoSInt<int> c1, int d);
	CoSInt<int> operator-(int d, CoUInt<int> c1);

	CoUInt<unsigned> operator*(CoSInt<int> c1, CoSInt<int> c2);
	CoUInt<unsigned> operator*(CoSInt<int> c1, int d);
	CoUInt<unsigned> operator*(int d, CoSInt<int> c1);

	CoSInt<int> operator/(CoSInt<int> c1, CoSInt<int> c2);
	CoSInt<int> operator/(CoSInt<int> c1, int d);
	CoSInt<int> operator/(int d, CoSInt<int> c1);

	bool operator==(CoSInt<int> c1, CoSInt<int> c2);
	bool operator!=(CoSInt<int> c1, CoSInt<int> c2);

	//istream& operator>>(istream&, CoSInt<int>& c1); //input
	//ostream& operator<<(ostream&, CoSInt<int>& c1); //output

	//end CoSInt function declarations

	template<> class CoUInt<unsigned> {
	public:
		CoUInt(unsigned val);
		unsigned val() const;
		// member operator overloads
		CoUInt<unsigned>& operator+=(CoUInt<unsigned> c)
		CoUInt<unsigned>& operator+=(unsigned c)
		CoUInt<unsigned>& operator-=(CoUInt<unsigned> c)
		CoUInt<unsigned>& operator-=(unsigned c)
		CoUInt<unsigned>& operator*=(CoUInt<unsigned> c)
		CoUInt<unsigned>& operator*=(unsigned c)
		CoUInt<unsigned>& operator/=(CoUInt<unsigned> c)
		CoUInt<unsigned>& operator/=(unsigned c)
	private:
		unsigned value;
	};

	//start CoUInt function declarations
	
	CoUInt<unsigned> operator+(CoUInt<unsigned> c1, CoUInt<unsigned> c2);
	CoUInt<unsigned> operator+(CoUInt<unsigned> c1, unsigned d);
	CoUInt<unsigned> operator+(unsigned d, CoCDouble<cufftDoubleComplex> c1);

	CoUInt<unsigned> operator-(CoUInt<unsigned> c1, CoUInt<unsigned> c2);
	CoUInt<unsigned> operator-(CoUInt<unsigned> c1, unsigned d);
	CoUInt<unsigned> operator-(unsigned d, CoUInt<unsigned> c1);

	CoUInt<unsigned> operator*(CoUInt<unsigned> c1, CoUInt<unsigned> c2);
	CoUInt<unsigned> operator*(CoUInt<unsigned> c1, double d);
	CoUInt<unsigned> operator*(double d, CoUInt<unsigned> c1);

	CoUInt<unsigned> operator/(CoUInt<unsigned> c1, CoUInt<unsigned> c2);
	CoUInt<unsigned> operator/(CoUInt<unsigned> c1, double d);
	CoUInt<unsigned> operator/(double d, CoUInt<unsigned> c1);

	bool operator==(CoUInt<unsigned> c1, CoUInt<unsigned> c2);
	bool operator!=(CoUInt<unsigned> c1, CoUInt<unsigned> c2);

	//istream& operator>>(istream&, CoUInt<unsigned>& c1); //input
	//ostream& operator<<(ostream&, CoUInt<unsigned>& c1); //output

	//end CoUInt function declarations
	
	typedef CoBool<bool> Bool;
	typedef CoFloat<float> Float;
	typedef CoCFloat<cufftComplex> CFloat;
	typedef CoDouble<double> Double;
	typedef CoCDouble<cufftDoubleComplex> CDouble;
	typedef CoSInt<int> Int;
	typedef CoUInt<unsigned> UInt;
}

#endif
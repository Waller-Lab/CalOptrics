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
	class CoBool {
	public:
		CoBool(bool val);
		bool val() const;
		// member operator overloads
		
	private:
		bool value;
	};

	//start CoBool function declarations
	
	CoBool operator&&(CoBool c1, CoBool c2);
	CoBool operator&&(CoBool c1, bool d);
	CoBool operator&&(bool d, CoBool c1);

	CoBool operator||(CoBool c1, CoBool c2);
	CoBool operator||(CoBool c1, bool d);
	CoBool operator||(bool d, CoBool c1);

	CoBool operator!(CoBool c1);

	bool operator==(CoBool c1, CoBool c2);
	bool operator!=(CoBool c1, CoBool c2);

	//istream& operator>>(istream&, CoBool& c1); //input
	//ostream& operator<<(ostream&, CoBool& c1); //output

	//end CoBool function declarations

	class CoFloat 
	{
	public:
		CoFloat(float val);
		float val() const;
		// member operator overloads
		CoFloat& operator+=(CoFloat c);
		CoFloat& operator+=(float c);
		CoFloat& operator-=(CoFloat c);
		CoFloat& operator-=(float c);
		CoFloat& operator*=(CoFloat c);
		CoFloat& operator*=(float c);
		CoFloat& operator/=(CoFloat c);
		CoFloat& operator/=(float c);
	private:
		float value;
	};

	//start CoFloat function declarations
	
	CoFloat operator+(CoFloat c1, CoFloat c2);
	CoFloat operator+(CoFloat c1, float d);
	CoFloat operator+(float d, CoFloat c1);

	CoFloat operator-(CoFloat c1, CoFloat c2);
	CoFloat operator-(CoFloat c1, float d);
	CoFloat operator-(float d, CoFloat c1);

	CoFloat operator*(CoFloat c1, CoFloat c2);
	CoFloat operator*(CoFloat c1, float d);
	CoFloat operator*(float d, CoFloat c1);

	CoFloat operator/(CoFloat c1, CoFloat c2);
	CoFloat operator/(CoFloat c1, float d);
	CoFloat operator/(float d, CoFloat c1);

	bool operator==(CoFloat c1, CoFloat c2);
	bool operator!=(CoFloat c1, CoFloat c2);

	//istream& operator>>(istream&, CoFloat& c1); //input
	//ostream& operator<<(ostream&, CoFloat& c1); //output

	//end CoFloat function declarations

	class CoCFloat {
	public:
		CoCFloat(cufftComplex val);
		cufftComplex val() const;
		// member operator overloads
		CoCFloat& operator+=(CoCFloat c);
		CoCFloat& operator+=(cufftComplex c);
		CoCFloat& operator+=(float c);
		CoCFloat& operator-=(CoCFloat c);
		CoCFloat& operator-=(cufftComplex c);
		CoCFloat& operator-=(float c);
		CoCFloat& operator*=(CoCFloat c);
		CoCFloat& operator*=(cufftComplex c);
		CoCFloat& operator*=(float c);
		CoCFloat& operator/=(CoCFloat c);
		CoCFloat& operator/=(cufftComplex c);
		CoCFloat& operator/=(float c);
	private:
		cufftComplex value;
	};

	//start CoCFloat function declarations
	
	CoCFloat operator+(CoCFloat c1, CoCFloat c2);
	CoCFloat operator+(CoCFloat c1, float d);
	CoCFloat operator+(float d, CoCFloat c1);

	CoCFloat operator-(CoCFloat c1, CoCFloat c2);
	CoCFloat operator-(CoCFloat c1, float d);
	CoCFloat operator-(float d, CoCFloat c1);

	CoCFloat operator*(CoCFloat c1, CoCFloat c2);
	CoCFloat operator*(CoCFloat c1, float d);
	CoCFloat operator*(float d, CoCFloat c1);

	CoCFloat operator/(CoCFloat c1, CoCFloat c2);
	CoCFloat operator/(CoCFloat c1, float d);
	CoCFloat operator/(float d, CoCFloat c1);

	bool operator==(CoCFloat c1, CoCFloat c2);
	bool operator!=(CoCFloat c1, CoCFloat c2);

	//istream& operator>>(istream&, CoCFloat& c1); //input
	//ostream& operator<<(ostream&, CoCFloat& c1); //output

	CoCFloat polar(float rho, float theta);
	CoCFloat conj(CoCFloat c);

	float abs(CoCFloat c);
	float arg(CoCFloat c);
	float norm(CoCFloat c);

	float real(CoCFloat c);
	float imag(CoCFloat c);

	//end CoCFloat function declarations

	class CoDouble {
	public:
		CoDouble(double val);
		double val() const;
		CoDouble& operator+=(CoDouble c);
		CoDouble& operator+=(double c);
		CoDouble& operator-=(CoDouble c);
		CoDouble& operator-=(double c);
		CoDouble& operator*=(CoDouble c);
		CoDouble& operator*=(double c);
		CoDouble& operator/=(CoDouble c);
		CoDouble& operator/=(double c);
	private:
		double value;
	};

	//start CoDouble function declarations
	
	CoDouble operator+(CoDouble c1, CoDouble c2);
	CoDouble operator+(CoDouble c1, double d);
	CoDouble operator+(double d, CoDouble c1);

	CoDouble operator-(CoDouble c1, CoDouble c2);
	CoDouble operator-(CoDouble c1, double d);
	CoDouble operator-(double d, CoDouble c1);

	CoDouble operator*(CoDouble c1, CoDouble c2);
	CoDouble operator*(CoDouble c1, double d);
	CoDouble operator*(double d, CoDouble c1);

	CoDouble operator/(CoDouble c1, CoDouble c2);
	CoDouble operator/(CoDouble c1, double d);
	CoDouble operator/(double d, CoDouble c1);

	bool operator==(CoDouble c1, CoDouble c2);
	bool operator!=(CoDouble c1, CoDouble c2);

	//istream& operator>>(istream&, CoDouble& c1); //input
	//ostream& operator<<(ostream&, CoDouble& c1); //output

	//end CoDouble function declarations

	class CoCDouble {
	public:
		CoCDouble(cufftDoubleComplex val);
		cufftDoubleComplex val() const;
		// member operator overloads
		CoCDouble& operator+=(CoCDouble c);
		CoCDouble& operator+=(cufftDoubleComplex c);
		CoCDouble& operator+=(double c);
		CoCDouble& operator-=(CoCDouble c);
		CoCDouble& operator-=(cufftDoubleComplex c);
		CoCDouble& operator-=(double c);
		CoCDouble& operator*=(CoCDouble c);
		CoCDouble& operator*=(cufftDoubleComplex c);
		CoCDouble& operator*=(double c);
		CoCDouble& operator/=(CoCDouble c);
		CoCDouble& operator/=(cufftDoubleComplex c);
		CoCDouble& operator/=(double c);
	private:
		cufftDoubleComplex value;
	};

	//start CoCDouble function declarations
	
	CoCDouble operator+(CoCDouble c1, CoCDouble c2);
	CoCDouble operator+(CoCDouble c1, double d);
	CoCDouble operator+(double d, CoCDouble c1);

	CoCDouble operator-(CoCDouble c1, CoCDouble c2);
	CoCDouble operator-(CoCDouble c1, double d);
	CoCDouble operator-(double d, CoCDouble c1);

	CoCDouble operator*(CoCDouble c1, CoCDouble c2);
	CoCDouble operator*(CoCDouble c1, double d);
	CoCDouble operator*(double d, CoCDouble c1);

	CoCDouble operator/(CoCDouble c1, CoCDouble c2);
	CoCDouble operator/(CoCDouble c1, double d);
	CoCDouble operator/(double d, CoCDouble c1);

	bool operator==(CoCDouble c1, CoCDouble c2);
	bool operator!=(CoCDouble c1, CoCDouble c2);

	//istream& operator>>(istream&, CoCDouble& c1); //input
	//ostream& operator<<(ostream&, CoCDouble& c1); //output

	CoCDouble polar(double rho, double theta);
	CoCDouble conj(CoCDouble c);

	double abs(CoCDouble c);
	double arg(CoCDouble c);
	double norm(CoCDouble c);

	double real(CoCDouble c);
	double imag(CoCDouble c);

	//end CoCDouble function declarations

	class CoSInt {
	public:
		CoSInt(int val);
		int val() const;
		// member operator overloads
		CoSInt& operator+=(CoSInt c);
		CoSInt& operator+=(int c);
		CoSInt& operator-=(CoSInt c);
		CoSInt& operator-=(int c);
		CoSInt& operator*=(CoSInt c);
		CoSInt& operator*=(int c);
		CoSInt& operator/=(CoSInt c);
		CoSInt& operator/=(int c);
	private:
		int value;
	};

	//start CoSInt function declarations
	
	CoSInt operator+(CoSInt c1, CoSInt c2);
	CoSInt operator+(CoSInt c1, int d);
	CoSInt operator+(int d, CoSInt c1);

	CoSInt operator-(CoSInt c1, CoSInt c2);
	CoSInt operator-(CoSInt c1, int d);
	CoSInt operator-(int d, CoSInt c1);

	CoSInt operator*(CoSInt c1, CoSInt c2);
	CoSInt operator*(CoSInt c1, int d);
	CoSInt operator*(int d, CoSInt c1);

	CoSInt operator/(CoSInt c1, CoSInt c2);
	CoSInt operator/(CoSInt c1, int d);
	CoSInt operator/(int d, CoSInt c1);

	bool operator==(CoSInt c1, CoSInt c2);
	bool operator!=(CoSInt c1, CoSInt c2);

	//istream& operator>>(istream&, CoSInt& c1); //input
	//ostream& operator<<(ostream&, CoSInt& c1); //output

	//end CoSInt function declarations

	class CoUInt {
	public:
		CoUInt(unsigned val);
		unsigned val() const;
		// member operator overloads
		CoUInt& operator+=(CoUInt c);
		CoUInt& operator+=(unsigned c);
		CoUInt& operator-=(CoUInt c);
		CoUInt& operator-=(unsigned c);
		CoUInt& operator*=(CoUInt c);
		CoUInt& operator*=(unsigned c);
		CoUInt& operator/=(CoUInt c);
		CoUInt& operator/=(unsigned c);
	private:
		unsigned value;
	};

	//start CoUInt function declarations
	
	CoUInt operator+(CoUInt c1, CoUInt c2);
	CoUInt operator+(CoUInt c1, unsigned d);
	CoUInt operator+(unsigned d, CoUInt c1);

	CoUInt operator-(CoUInt c1, CoUInt c2);
	CoUInt operator-(CoUInt c1, unsigned d);
	CoUInt operator-(unsigned d, CoUInt c1);

	CoUInt operator*(CoUInt c1, CoUInt c2);
	CoUInt operator*(CoUInt c1, unsigned d);
	CoUInt operator*(unsigned d, CoUInt c1);

	CoUInt operator/(CoUInt c1, CoUInt c2);
	CoUInt operator/(CoUInt c1, unsigned d);
	CoUInt operator/(unsigned d, CoUInt c1);

	bool operator==(CoUInt c1, CoUInt c2);
	bool operator!=(CoUInt c1, CoUInt c2);

	//istream& operator>>(istream&, CoUInt& c1); //input
	//ostream& operator<<(ostream&, CoUInt& c1); //output

	//end CoUInt function declarations
	
	typedef CoBool Bool;
	typedef CoFloat Float;
	typedef CoCFloat CFloat;
	typedef CoDouble Double;
	typedef CoCDouble CDouble;
	typedef CoSInt Int;
	typedef CoUInt UInt;
}

#endif
//	co_datatypes.cpp
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

#include <cufft.h>
#include <cmath>
#include <iostream>
#include "co_datatypes.h"

namespace co {

	//constructor member functions
	CoCDouble::CoCDouble(cufftDoubleComplex val)
	{
		this->value = val;
	}

	CoSInt::CoSInt(int val)
	{
		this->value = val;
	}

	CoDouble::CoDouble(double val)
	{
		this->value = val;
	}

	CoFloat::CoFloat(float val)
	{
		this->value = val;
	}

	CoUInt::CoUInt(unsigned val)
	{
		this->value = val;
	}

	CoBool::CoBool(bool val)
	{
		this->value = val;
	}

	CoCFloat::CoCFloat(cufftComplex val)
	{
		this->value = val;
	}

	//getter member functions
	cufftDoubleComplex CoCDouble::val() const
	{
		return this->value;
	}

	int CoSInt::val() const
	{
		return this->value;
	}

	double CoDouble::val() const
	{
		return this->value;
	}

	float CoFloat::val() const
	{
		return this->value;
	}

	unsigned CoUInt::val() const
	{
		return this->value;
	}

	bool CoBool::val() const
	{
		return this->value;
	}

	cufftComplex CoCFloat::val() const
	{
		return this->value;
	}

	//member operator overloads for non-complex data types
	CoSInt& CoSInt::operator-=(CoSInt other)
	{
		this->value -= other.val();
		return *this;
	}

	CoSInt& CoSInt::operator-=(int other)
	{
		this->value -= other;
		return *this;
	}

	CoSInt& CoSInt::operator*=(CoSInt other)
	{
		this->value *= other.val();
		return *this;
	}

	CoSInt& CoSInt::operator*=(int other)
	{
		this->value *= other;
		return *this;
	}

	CoSInt& CoSInt::operator/=(CoSInt other)
	{
		this->value /= other.val();
		return *this;
	}

	CoSInt& CoSInt::operator/=(int other)
	{
		this->value /= other;
		return *this;
	}

	CoSInt& CoSInt::operator+=(CoSInt other)
	{
		this->value += other.val();
		return *this;
	}

	CoSInt& CoSInt::operator+=(int other)
	{
		this->value += other;
		return *this;
	}

	CoDouble& CoDouble::operator-=(CoDouble other)
	{
		this->value -= other.val();
		return *this;
	}

	CoDouble& CoDouble::operator-=(double other)
	{
		this->value -= other;
		return *this;
	}

	CoDouble& CoDouble::operator*=(CoDouble other)
	{
		this->value *= other.val();
		return *this;
	}

	CoDouble& CoDouble::operator*=(double other)
	{
		this->value *= other;
		return *this;
	}

	CoDouble& CoDouble::operator/=(CoDouble other)
	{
		this->value /= other.val();
		return *this;
	}

	CoDouble& CoDouble::operator/=(double other)
	{
		this->value /= other;
		return *this;
	}

	CoDouble& CoDouble::operator+=(CoDouble other)
	{
		this->value += other.val();
		return *this;
	}

	CoDouble& CoDouble::operator+=(double other)
	{
		this->value += other;
		return *this;
	}

	CoFloat& CoFloat::operator-=(CoFloat other)
	{
		this->value -= other.val();
		return *this;
	}

	CoFloat& CoFloat::operator-=(float other)
	{
		this->value -= other;
		return *this;
	}

	CoFloat& CoFloat::operator*=(CoFloat other)
	{
		this->value *= other.val();
		return *this;
	}

	CoFloat& CoFloat::operator*=(float other)
	{
		this->value *= other;
		return *this;
	}

	CoFloat& CoFloat::operator/=(CoFloat other)
	{
		this->value /= other.val();
		return *this;
	}

	CoFloat& CoFloat::operator/=(float other)
	{
		this->value /= other;
		return *this;
	}

	CoFloat& CoFloat::operator+=(CoFloat other)
	{
		this->value += other.val();
		return *this;
	}

	CoFloat& CoFloat::operator+=(float other)
	{
		this->value += other;
		return *this;
	}

	CoUInt& CoUInt::operator-=(CoUInt other)
	{
		this->value -= other.val();
		return *this;
	}

	CoUInt& CoUInt::operator-=(unsigned other)
	{
		this->value -= other;
		return *this;
	}

	CoUInt& CoUInt::operator*=(CoUInt other)
	{
		this->value *= other.val();
		return *this;
	}

	CoUInt& CoUInt::operator*=(unsigned other)
	{
		this->value *= other;
		return *this;
	}

	CoUInt& CoUInt::operator/=(CoUInt other)
	{
		this->value /= other.val();
		return *this;
	}

	CoUInt& CoUInt::operator/=(unsigned other)
	{
		this->value /= other;
		return *this;
	}

	CoUInt& CoUInt::operator+=(CoUInt other)
	{
		this->value += other.val();
		return *this;
	}

	CoUInt& CoUInt::operator+=(unsigned other)
	{
		this->value += other;
		return *this;
	}

	//member operator overloads for complex data types
	CoCDouble& CoCDouble::operator-=(CoCDouble other)
	{
		this->value.x -= other.val().x;
		this->value.y -= other.val().y;
		return *this;
	}

	CoCDouble& CoCDouble::operator-=(cufftDoubleComplex other)
	{
		this->value.x -= other.x;
		this->value.y -= other.y;
		return *this;
	}

	CoCDouble& CoCDouble::operator-=(double other)
	{
		this->value.x -= other;
		return *this;
	}

	CoCDouble& CoCDouble::operator*=(CoCDouble other)
	{
		this->value.x = this->value.x*other.val().x - this->value.y*other.val().y;
		this->value.y = this->value.x*other.val().y + this->value.y*other.val().x;
		return *this;
	}

	CoCDouble& CoCDouble::operator*=(cufftDoubleComplex other)
	{
		this->value.x = this->value.x*other.x - this->value.y*other.y;
		this->value.y = this->value.x*other.y + this->value.y*other.x;
		return *this;
	}

	CoCDouble& CoCDouble::operator*=(double other)
	{
		this->value.x *= other;
		this->value.y *= other;
		return *this;
	}

	CoCDouble& CoCDouble::operator/=(CoCDouble other)
	{
		this->value.x = this->value.x*other.val().x - this->value.y*other.val().y;
		this->value.y = this->value.x*other.val().y + this->value.y*other.val().x;
		this->value.x *= 1/norm(other);
		this->value.y *= 1/norm(other);
		return *this;
	}

	CoCDouble& CoCDouble::operator/=(cufftDoubleComplex other)
	{
		CoCDouble tmp = CoCDouble(other);
		this->value.x = this->value.x*tmp.val().x - this->value.y*tmp.val().y;
		this->value.y = this->value.x*tmp.val().y + this->value.y*tmp.val().x;
		this->value.x *= 1/norm(tmp);
		this->value.y *= 1/norm(tmp);
		return *this;
	}

	CoCDouble& CoCDouble::operator/=(double other)
	{
		this->value.x /= other;
		this->value.y /= other;
		return *this;
	}

	CoCDouble& CoCDouble::operator+=(CoCDouble other)
	{
		this->value.x += other.val().x;
		this->value.y += other.val().y;
		return *this;
	}

	CoCDouble& CoCDouble::operator+=(cufftDoubleComplex other)
	{
		this->value.x += other.x;
		this->value.y += other.y;
		return *this;
	}

	CoCDouble& CoCDouble::operator+=(double other)
	{
		this->value.x += other;
		return *this;
	}

	CoCFloat& CoCFloat::operator-=(CoCFloat other)
	{
		this->value.x -= other.val().x;
		this->value.y -= other.val().y;
		return *this;
	}

	CoCFloat& CoCFloat::operator-=(cufftComplex other)
	{
		this->value.x -= other.x;
		this->value.y -= other.y;
		return *this;
	}

	CoCFloat& CoCFloat::operator-=(float other)
	{
		this->value.x -= other;
		return *this;
	}

	CoCFloat& CoCFloat::operator*=(CoCFloat other)
	{
		this->value.x = this->value.x*other.val().x - this->value.y*other.val().y;
		this->value.y = this->value.x*other.val().y + this->value.y*other.val().x;
		return *this;
	}

	CoCFloat& CoCFloat::operator*=(cufftComplex other)
	{
		this->value.x = this->value.x*other.x - this->value.y*other.y;
		this->value.y = this->value.x*other.y + this->value.y*other.x;
		return *this;
	}

	CoCFloat& CoCFloat::operator*=(float other)
	{
		this->value.x *= other;
		this->value.y *= other;
		return *this;
	}

	CoCFloat& CoCFloat::operator/=(CoCFloat other)
	{
		this->value.x = this->value.x*other.val().x - this->value.y*other.val().y;
		this->value.y = this->value.x*other.val().y + this->value.y*other.val().x;
		this->value.x *= 1/norm(other);
		this->value.y *= 1/norm(other);
		return *this;
	}

	CoCFloat& CoCFloat::operator/=(cufftComplex other)
	{
		CoCFloat tmp = CoCFloat(other);
		this->value.x = this->value.x*tmp.val().x - this->value.y*tmp.val().y;
		this->value.y = this->value.x*tmp.val().y + this->value.y*tmp.val().x;
		this->value.x *= 1/norm(tmp);
		this->value.y *= 1/norm(tmp);
		return *this;
	}

	CoCFloat& CoCFloat::operator/=(float other)
	{
		this->value.x /= other;
		this->value.y /= other;
		return *this;
	}

	CoCFloat& CoCFloat::operator+=(CoCFloat other)
	{
		this->value.x += other.val().x;
		this->value.y += other.val().y;
		return *this;
	}

	CoCFloat& CoCFloat::operator+=(cufftComplex other)
	{
		this->value.x += other.x;
		this->value.y += other.y;
		return *this;
	}

	CoCFloat& CoCFloat::operator+=(float other)
	{
		this->value.x += other;
		return *this;
	}

	//create nonmember arithmetic,equality, and IO operator overloads for real data types
	CoSInt operator+(CoSInt a,CoSInt b)
	{
		CoSInt r = a;
		return r += b;
	}

	CoSInt operator+(CoSInt a,int b)
	{
		CoSInt r = a;
		return r += b;
	}

	CoSInt operator+(int a,CoSInt b)
	{
		CoSInt r = b;
		return r += a;
	}

	CoSInt operator*(CoSInt a,CoSInt b)
	{
		CoSInt r = a;
		return r *= b;
	}

	CoSInt operator*(CoSInt a,int b)
	{
		CoSInt r = a;
		return r *= b;
	}

	CoSInt operator*(int a,CoSInt b)
	{
		CoSInt r = b;
		return r *= a;
	}

	CoSInt operator-(CoSInt a,CoSInt b)
	{
		CoSInt r = a;
		return r -= b;
	}

	CoSInt operator-(CoSInt a,int b)
	{
		CoSInt r = a;
		return r -= b;
	}

	CoSInt operator-(int a,CoSInt b)
	{
		CoSInt r = b;
		return r -= a;
	}

	CoSInt operator/(CoSInt a,CoSInt b)
	{
		CoSInt r = a;
		return r /= b;
	}

	CoSInt operator/(CoSInt a,int b)
	{
		CoSInt r = a;
		return r /= b;
	}

	CoSInt operator/(int a,CoSInt b)
	{
		CoSInt r = b;
		return r /= a;
	}

	bool operator==(CoSInt a,CoSInt b)
	{
		return a.val() == b.val();
	}

	bool operator!=(CoSInt a,CoSInt b)
	{
		return a.val() != b.val();
	}

	CoDouble operator+(CoDouble a,CoDouble b)
	{
		CoDouble r = a;
		return r += b;
	}

	CoDouble operator+(CoDouble a,double b)
	{
		CoDouble r = a;
		return r += b;
	}

	CoDouble operator+(double a,CoDouble b)
	{
		CoDouble r = b;
		return r += a;
	}

	CoDouble operator*(CoDouble a,CoDouble b)
	{
		CoDouble r = a;
		return r *= b;
	}

	CoDouble operator*(CoDouble a,double b)
	{
		CoDouble r = a;
		return r *= b;
	}

	CoDouble operator*(double a,CoDouble b)
	{
		CoDouble r = b;
		return r *= a;
	}

	CoDouble operator-(CoDouble a,CoDouble b)
	{
		CoDouble r = a;
		return r -= b;
	}

	CoDouble operator-(CoDouble a,double b)
	{
		CoDouble r = a;
		return r -= b;
	}

	CoDouble operator-(double a,CoDouble b)
	{
		CoDouble r = b;
		return r -= a;
	}

	CoDouble operator/(CoDouble a,CoDouble b)
	{
		CoDouble r = a;
		return r /= b;
	}

	CoDouble operator/(CoDouble a,double b)
	{
		CoDouble r = a;
		return r /= b;
	}

	CoDouble operator/(double a,CoDouble b)
	{
		CoDouble r = b;
		return r /= a;
	}

	bool operator==(CoDouble a,CoDouble b)
	{
		return a.val() == b.val();
	}

	bool operator!=(CoDouble a,CoDouble b)
	{
		return a.val() != b.val();
	}

	CoFloat operator+(CoFloat a,CoFloat b)
	{
		CoFloat r = a;
		return r += b;
	}

	CoFloat operator+(CoFloat a,float b)
	{
		CoFloat r = a;
		return r += b;
	}

	CoFloat operator+(float a,CoFloat b)
	{
		CoFloat r = b;
		return r += a;
	}

	CoFloat operator*(CoFloat a,CoFloat b)
	{
		CoFloat r = a;
		return r *= b;
	}

	CoFloat operator*(CoFloat a,float b)
	{
		CoFloat r = a;
		return r *= b;
	}

	CoFloat operator*(float a,CoFloat b)
	{
		CoFloat r = b;
		return r *= a;
	}

	CoFloat operator-(CoFloat a,CoFloat b)
	{
		CoFloat r = a;
		return r -= b;
	}

	CoFloat operator-(CoFloat a,float b)
	{
		CoFloat r = a;
		return r -= b;
	}

	CoFloat operator-(float a,CoFloat b)
	{
		CoFloat r = b;
		return r -= a;
	}

	CoFloat operator/(CoFloat a,CoFloat b)
	{
		CoFloat r = a;
		return r /= b;
	}

	CoFloat operator/(CoFloat a,float b)
	{
		CoFloat r = a;
		return r /= b;
	}

	CoFloat operator/(float a,CoFloat b)
	{
		CoFloat r = b;
		return r /= a;
	}

	bool operator==(CoFloat a,CoFloat b)
	{
		return a.val() == b.val();
	}

	bool operator!=(CoFloat a,CoFloat b)
	{
		return a.val() != b.val();
	}

	CoUInt operator+(CoUInt a,CoUInt b)
	{
		CoUInt r = a;
		return r += b;
	}

	CoUInt operator+(CoUInt a,unsigned b)
	{
		CoUInt r = a;
		return r += b;
	}

	CoUInt operator+(unsigned a,CoUInt b)
	{
		CoUInt r = b;
		return r += a;
	}

	CoUInt operator*(CoUInt a,CoUInt b)
	{
		CoUInt r = a;
		return r *= b;
	}

	CoUInt operator*(CoUInt a,unsigned b)
	{
		CoUInt r = a;
		return r *= b;
	}

	CoUInt operator*(unsigned a,CoUInt b)
	{
		CoUInt r = b;
		return r *= a;
	}

	CoUInt operator-(CoUInt a,CoUInt b)
	{
		CoUInt r = a;
		return r -= b;
	}

	CoUInt operator-(CoUInt a,unsigned b)
	{
		CoUInt r = a;
		return r -= b;
	}

	CoUInt operator-(unsigned a,CoUInt b)
	{
		CoUInt r = b;
		return r -= a;
	}

	CoUInt operator/(CoUInt a,CoUInt b)
	{
		CoUInt r = a;
		return r /= b;
	}

	CoUInt operator/(CoUInt a,unsigned b)
	{
		CoUInt r = a;
		return r /= b;
	}

	CoUInt operator/(unsigned a,CoUInt b)
	{
		CoUInt r = b;
		return r /= a;
	}

	bool operator==(CoUInt a,CoUInt b)
	{
		return a.val() == b.val();
	}

	bool operator!=(CoUInt a,CoUInt b)
	{
		return a.val() != b.val();
	}

	CoBool operator&&(CoBool a,CoBool b)
	{
		return CoBool(a.val() && b.val());
	}

	CoBool operator&&(CoBool a,bool b)
	{
		
		return CoBool(a.val() && b);
	}

	CoBool operator&&(bool a,CoBool b)
	{
		return CoBool(a && b.val());
	}

	CoBool operator||(CoBool a,CoBool b)
	{
		return CoBool(a.val() || b.val());
	}

	CoBool operator||(CoBool a,bool b)
	{
		return CoBool(a.val() || b);
	}

	CoBool operator||(bool a,CoBool b)
	{
		return CoBool(a || b.val());
	}

	CoBool operator!(CoBool a)
	{
		return CoBool(!a.val());
	}

	bool operator==(CoBool a,CoBool b)
	{
		return a.val() == b.val();
	}

	bool operator!=(CoBool a,CoBool b)
	{
		return a.val() != b.val();
	}

	//create nonmember arithmetic,equality, and IO operator overloads for complex data types
	CoCDouble operator+(CoCDouble a,CoCDouble b)
	{
		CoCDouble r = a;
		return r += b;
	}

	CoCDouble operator+(CoCDouble a,cufftDoubleComplex b)
	{
		CoCDouble r = a;
		return r += b;
	}

	CoCDouble operator+(cufftDoubleComplex a,CoCDouble b)
	{
		CoCDouble r = b;
		return r += a;
	}

	CoCDouble operator*(CoCDouble a,CoCDouble b)
	{
		CoCDouble r = a;
		return r *= b;
	}

	CoCDouble operator*(CoCDouble a,cufftDoubleComplex b)
	{
		CoCDouble r = a;
		return r *= b;
	}

	CoCDouble operator*(cufftDoubleComplex a,CoCDouble b)
	{
		CoCDouble r = b;
		return r *= a;
	}

	CoCDouble operator-(CoCDouble a,CoCDouble b)
	{
		CoCDouble r = a;
		return r -= b;
	}

	CoCDouble operator-(CoCDouble a,cufftDoubleComplex b)
	{
		CoCDouble r = a;
		return r -= b;
	}

	CoCDouble operator-(cufftDoubleComplex a,CoCDouble b)
	{
		CoCDouble r = b;
		return r -= a;
	}

	CoCDouble operator/(CoCDouble a,CoCDouble b)
	{
		CoCDouble r = a;
		return r /= b;
	}

	CoCDouble operator/(CoCDouble a,cufftDoubleComplex b)
	{
		CoCDouble r = a;
		return r /= b;
	}

	CoCDouble operator/(cufftDoubleComplex a,CoCDouble b)
	{
		CoCDouble r = b;
		return r /= a;
	}

	bool operator==(CoCDouble a,CoCDouble b)
	{
		return a.val().x == b.val().x && a.val().y == b.val().y;
	}

	bool operator!=(CoCDouble a,CoCDouble b)
	{
		return a.val().x != b.val().x || a.val().y != b.val().y;
	}

	CoCFloat operator+(CoCFloat a,CoCFloat b)
	{
		CoCFloat r = a;
		return r += b;
	}

	CoCFloat operator+(CoCFloat a,cufftComplex b)
	{
		CoCFloat r = a;
		return r += b;
	}

	CoCFloat operator+(cufftComplex a,CoCFloat b)
	{
		CoCFloat r = b;
		return r += a;
	}

	CoCFloat operator*(CoCFloat a,CoCFloat b)
	{
		CoCFloat r = a;
		return r *= b;
	}

	CoCFloat operator*(CoCFloat a,cufftComplex b)
	{
		CoCFloat r = a;
		return r *= b;
	}

	CoCFloat operator*(cufftComplex a,CoCFloat b)
	{
		CoCFloat r = b;
		return r *= a;
	}

	CoCFloat operator-(CoCFloat a,CoCFloat b)
	{
		CoCFloat r = a;
		return r -= b;
	}

	CoCFloat operator-(CoCFloat a,cufftComplex b)
	{
		CoCFloat r = a;
		return r -= b;
	}

	CoCFloat operator-(cufftComplex a,CoCFloat b)
	{
		CoCFloat r = b;
		return r -= a;
	}

	CoCFloat operator/(CoCFloat a,CoCFloat b)
	{
		CoCFloat r = a;
		return r /= b;
	}

	CoCFloat operator/(CoCFloat a,cufftComplex b)
	{
		CoCFloat r = a;
		return r /= b;
	}

	CoCFloat operator/(cufftComplex a,CoCFloat b)
	{
		CoCFloat r = b;
		return r /= a;
	}

	bool operator==(CoCFloat a,CoCFloat b)
	{
		return a.val().x == b.val().x && a.val().y == b.val().y;
	}

	bool operator!=(CoCFloat a,CoCFloat b)
	{
		return a.val().x != b.val().x || a.val().y != b.val().y;
	}

	//Complex utility functions
	//float
	CoCFloat polar(float rho, float theta)
	{
		cufftComplex c;
		c.x = rho*std::cos(theta);
		c.y = rho*std::sin(theta);
		return CoCFloat(c);
	}
	CoCFloat conj(CoCFloat c)
	{
		cufftComplex b;
		b.x = c.val().x;
		b.y = -c.val().y;
		return CoCFloat(b);
	}

	float abs(CoCFloat c)
	{
		return std::sqrt(real(c)*real(c) + imag(c)*imag(c));
	}

	float arg(CoCFloat c)
	{
		return std::atan2(c.val().y, c.val().x);
	}

	float norm(CoCFloat c)
	{
		return std::sqrt(real(c)*real(c) + imag(c)*imag(c));
	}

	float real(CoCFloat c)
	{
		return c.val().x;
	}

	float imag(CoCFloat c)
	{
		return c.val().y;
	}

	//double
	CoCDouble polar(double rho, double theta)
	{
		cufftDoubleComplex c;
		c.x = rho*std::cos(theta);
		c.y = rho*std::sin(theta);
		return CoCDouble(c);
	}
	CoCDouble conj(CoCDouble c)
	{
		cufftDoubleComplex b;
		b.x = c.val().x;
		b.y = -c.val().y;
		return CoCDouble(b);
	}

	double abs(CoCDouble c)
	{
		return std::sqrt(real(c)*real(c) + imag(c)*imag(c));
	}

	double arg(CoCDouble c)
	{
		return std::atan2(c.val().y,c.val().x);
	}

	double norm(CoCDouble c)
	{
		return std::sqrt(real(c)*real(c) + imag(c)*imag(c));
	}

	double real(CoCDouble c)
	{
		return c.val().x;
	}

	double imag(CoCDouble c)
	{
		return c.val().y;
	}
}
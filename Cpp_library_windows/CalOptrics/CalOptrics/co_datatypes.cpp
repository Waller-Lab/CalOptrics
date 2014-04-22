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
	CoCDouble<cufftDoubleComplex>::CoCDouble(cufftDoubleComplex val)
	{
		this->value = val;
	}

	CoSInt<int>::CoSInt(int val)
	{
		this->value = val;
	}

	CoDouble<double>::CoDouble(double val)
	{
		this->value = val;
	}

	CoFloat<float>::CoFloat(float val)
	{
		this->value = val;
	}

	CoUInt<unsigned>::CoUInt(unsigned val)
	{
		this->value = val;
	}

	CoBool<bool>::CoBool(bool val)
	{
		this->value = val;
	}

	CoCFloat<cufftComplex>::CoCFloat(cufftComplex val)
	{
		this->value = val;
	}

	//getter member functions
	cufftDoubleComplex CoCDouble<cufftDoubleComplex>::val() const
	{
		return this->val;
	}

	int CoSInt<int>::val() const
	{
		return this->val;
	}

	double CoDouble<double>::val() const
	{
		return this->val;
	}

	float CoFloat<float>::val() const
	{
		return this->val;
	}

	unsigned CoUInt<unsigned>::val() const
	{
		return this->val;
	}

	bool CoBool<bool>::val() const
	{
		return this->val;
	}

	cufftComplex CoCFloat<cufftComplex>::val() const
	{
		return this->val;
	}

	//member operator overloads for non-complex data types
	CoSInt<int>& CoSInt<int>::operator-=(CoSInt<int> other)
	{
		this->value -= other.val();
		return *this;
	}

	CoSInt<int>& CoSInt<int>::operator-=(int other)
	{
		this->value -= other;
		return *this;
	}

	CoSInt<int>& CoSInt<int>::operator*=(CoSInt<int> other)
	{
		this->value *= other.val();
		return *this;
	}

	CoSInt<int>& CoSInt<int>::operator*=(int other)
	{
		this->value *= other;
		return *this;
	}

	CoSInt<int>& CoSInt<int>::operator/=(CoSInt<int> other)
	{
		this->value /= other.val();
		return *this;
	}

	CoSInt<int>& CoSInt<int>::operator/=(int other)
	{
		this->value /= other;
		return *this;
	}

	CoSInt<int>& CoSInt<int>::operator+=(CoSInt<int> other)
	{
		this->value += other.val();
		return *this;
	}

	CoSInt<int>& CoSInt<int>::operator+=(int other)
	{
		this->value += other;
		return *this;
	}

	CoDouble<double>& CoDouble<double>::operator-=(CoDouble<double> other)
	{
		this->value -= other.val();
		return *this;
	}

	CoDouble<double>& CoDouble<double>::operator-=(double other)
	{
		this->value -= other;
		return *this;
	}

	CoDouble<double>& CoDouble<double>::operator*=(CoDouble<double> other)
	{
		this->value *= other.val();
		return *this;
	}

	CoDouble<double>& CoDouble<double>::operator*=(double other)
	{
		this->value *= other;
		return *this;
	}

	CoDouble<double>& CoDouble<double>::operator/=(CoDouble<double> other)
	{
		this->value /= other.val();
		return *this;
	}

	CoDouble<double>& CoDouble<double>::operator/=(double other)
	{
		this->value /= other;
		return *this;
	}

	CoDouble<double>& CoDouble<double>::operator+=(CoDouble<double> other)
	{
		this->value += other.val();
		return *this;
	}

	CoDouble<double>& CoDouble<double>::operator+=(double other)
	{
		this->value += other;
		return *this;
	}

	CoFloat<float>& CoFloat<float>::operator-=(CoFloat<float> other)
	{
		this->value -= other.val();
		return *this;
	}

	CoFloat<float>& CoFloat<float>::operator-=(float other)
	{
		this->value -= other;
		return *this;
	}

	CoFloat<float>& CoFloat<float>::operator*=(CoFloat<float> other)
	{
		this->value *= other.val();
		return *this;
	}

	CoFloat<float>& CoFloat<float>::operator*=(float other)
	{
		this->value *= other;
		return *this;
	}

	CoFloat<float>& CoFloat<float>::operator/=(CoFloat<float> other)
	{
		this->value /= other.val();
		return *this;
	}

	CoFloat<float>& CoFloat<float>::operator/=(float other)
	{
		this->value /= other;
		return *this;
	}

	CoFloat<float>& CoFloat<float>::operator+=(CoFloat<float> other)
	{
		this->value += other.val();
		return *this;
	}

	CoFloat<float>& CoFloat<float>::operator+=(float other)
	{
		this->value += other;
		return *this;
	}

	CoUInt<unsigned>& CoUInt<unsigned>::operator-=(CoUInt<unsigned> other)
	{
		this->value -= other.val();
		return *this;
	}

	CoUInt<unsigned>& CoUInt<unsigned>::operator-=(unsigned other)
	{
		this->value -= other;
		return *this;
	}

	CoUInt<unsigned>& CoUInt<unsigned>::operator*=(CoUInt<unsigned> other)
	{
		this->value *= other.val();
		return *this;
	}

	CoUInt<unsigned>& CoUInt<unsigned>::operator*=(unsigned other)
	{
		this->value *= other;
		return *this;
	}

	CoUInt<unsigned>& CoUInt<unsigned>::operator/=(CoUInt<unsigned> other)
	{
		this->value /= other.val();
		return *this;
	}

	CoUInt<unsigned>& CoUInt<unsigned>::operator/=(unsigned other)
	{
		this->value /= other;
		return *this;
	}

	CoUInt<unsigned>& CoUInt<unsigned>::operator+=(CoUInt<unsigned> other)
	{
		this->value += other.val();
		return *this;
	}

	CoUInt<unsigned>& CoUInt<unsigned>::operator+=(unsigned other)
	{
		this->value += other;
		return *this;
	}

	CoBool<bool>& CoBool<bool>::operator-=(CoBool<bool> other)
	{
		this->value -= other.val();
		return *this;
	}

	CoBool<bool>& CoBool<bool>::operator-=(bool other)
	{
		this->value -= other;
		return *this;
	}

	CoBool<bool>& CoBool<bool>::operator*=(CoBool<bool> other)
	{
		this->value *= other.val();
		return *this;
	}

	CoBool<bool>& CoBool<bool>::operator*=(bool other)
	{
		this->value *= other;
		return *this;
	}

	CoBool<bool>& CoBool<bool>::operator/=(CoBool<bool> other)
	{
		this->value /= other.val();
		return *this;
	}

	CoBool<bool>& CoBool<bool>::operator/=(bool other)
	{
		this->value /= other;
		return *this;
	}

	CoBool<bool>& CoBool<bool>::operator+=(CoBool<bool> other)
	{
		this->value += other.val();
		return *this;
	}

	CoBool<bool>& CoBool<bool>::operator+=(bool other)
	{
		this->value += other;
		return *this;
	}

	//member operator overloads for complex data types
	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex> CoCDouble<cufftDoubleComplex>::operator-=(CoCDouble<cufftDoubleComplex> other)
	{
		this->value.x -= other.val().x;
		this->value.y -= other.val().y;
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex>::operator-=(cufftDoubleComplex other)
	{
		this->value.x -= other.x;
		this->value.y -= other.y;
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex>::operator-=(double other)
	{
		this->value.x -= other;
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex> CoCDouble<cufftDoubleComplex>::operator*=(CoCDouble<cufftDoubleComplex> other)
	{
		this *= conj(other);
		this *= 1/norm(other);
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex>::operator*=(cufftDoubleComplex other)
	{
		CoCDouble<cufftDoubleComplex> tmp = CoCDouble(other);
		this *= conj(tmp);
		this *= 1/norm(tmp);
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex> CoCDouble<cufftDoubleComplex>::operator*=(double other)
	{
		this->value.x *= other;
		this->value.y *= other;
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex> CoCDouble<cufftDoubleComplex>::operator/=(CoCDouble<cufftDoubleComplex> other)
	{
		this->value.x /= other.val().x;
		this->value.y /= other.val().y;
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex>::operator/=(cufftDoubleComplex other)
	{
		this->value.x /= other.x;
		this->value.y /= other.y;
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex>::operator/=(double other)
	{
		this->value.x /= other;
		this->value.y /= other;
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex> CoCDouble<cufftDoubleComplex>::operator+=(CoCDouble<cufftDoubleComplex> other)
	{
		this->value.x += other.val().x;
		this->value.y += other.val().y;
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex>::operator+=(cufftDoubleComplex other)
	{
		this->value.x += other.x;
		this->value.y += other.y;
		return *this;
	}

	CoCDouble<cufftDoubleComplex>& CoCDouble<cufftDoubleComplex>::operator+=(double other)
	{
		this->value.x += other;
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex> CoCFloat<cufftComplex>::operator-=(CoCFloat<cufftComplex> other)
	{
		this->value.x -= other.val().x;
		this->value.y -= other.val().y;
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex>::operator-=(cufftComplex other)
	{
		this->value.x -= other.x;
		this->value.y -= other.y;
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex>::operator-=(float other)
	{
		this->value.x -= other;
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex> CoCFloat<cufftComplex>::operator*=(CoCFloat<cufftComplex> other)
	{
		this->value.x = this->value.x*other.val().x - this->value.y*other.val().y;
		this->value.y = this->value.x*other.val().y + this->value.y*other.val().x;
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex>::operator*=(cufftComplex other)
	{
		this->value.x = this->value.x*other.x - this->value.y*other.y;
		this->value.y = this->value.x*other.y + this->value.y*other.x;
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex>::operator*=(float other)
	{
		this->value.x *= other;
		this->value.y *= other;
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex> CoCFloat<cufftComplex>::operator/=(CoCFloat<cufftComplex> other)
	{
		this *= conj(other);
		this *= 1/norm(other);
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex>::operator/=(cufftComplex other)
	{
		CoCFloat<cufftComplex> tmp = CoCFloat(other);
		this *= conj(tmp);
		this *= 1/norm(tmp);
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex>::operator/=(float other)
	{
		this->value.x /= other;
		this->value.y /= other;
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex> CoCFloat<cufftComplex>::operator+=(CoCFloat<cufftComplex> other)
	{
		this->value.x += other.val().x;
		this->value.y += other.val().y;
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex>::operator+=(cufftComplex other)
	{
		this->value.x += other.x;
		this->value.y += other.y;
		return *this;
	}

	CoCFloat<cufftComplex>& CoCFloat<cufftComplex>::operator+=(float other)
	{
		this->value.x += other;
		return *this;
	}

	//create nonmember arithmetic,equality, and IO operator overloads for real data types
	CoSInt<int> operator+(CoSInt<int> a,CoSInt<int> b)
	{
		CoSInt<int> r = a;
		return r += b;
	}

	CoSInt<int> operator+(CoSInt<int> a,int b)
	{
		CoSInt<int> r = a;
		return r += b;
	}

	CoSInt<int> operator+(int a,CoSInt<int> b)
	{
		CoSInt<int> r = b;
		return r += a;
	}

	CoSInt<int> operator*(CoSInt<int> a,CoSInt<int> b)
	{
		CoSInt<int> r = a;
		return r *= b;
	}

	CoSInt<int> operator*(CoSInt<int> a,int b)
	{
		CoSInt<int> r = a;
		return r *= b;
	}

	CoSInt<int> operator*(int a,CoSInt<int> b)
	{
		CoSInt<int> r = b;
		return r *= a;
	}

	CoSInt<int> operator-(CoSInt<int> a,CoSInt<int> b)
	{
		CoSInt<int> r = a;
		return r -= b;
	}

	CoSInt<int> operator-(CoSInt<int> a,int b)
	{
		CoSInt<int> r = a;
		return r -= b;
	}

	CoSInt<int> operator-(int a,CoSInt<int> b)
	{
		CoSInt<int> r = b;
		return r -= a;
	}

	CoSInt<int> operator/(CoSInt<int> a,CoSInt<int> b)
	{
		CoSInt<int> r = a;
		return r /= b;
	}

	CoSInt<int> operator/(CoSInt<int> a,int b)
	{
		CoSInt<int> r = a;
		return r /= b;
	}

	CoSInt<int> operator/(int a,CoSInt<int> b)
	{
		CoSInt<int> r = b;
		return r /= a;
	}

	bool operator==(CoSInt<int> a,CoSInt<int> b)
	{
		return a.val() == b.val();
	}

	bool operator!=(CoSInt<int> a,CoSInt<int> b)
	{
		return a.val() != b.val();
	}

	CoDouble<double> operator+(CoDouble<double> a,CoDouble<double> b)
	{
		CoDouble<double> r = a;
		return r += b;
	}

	CoDouble<double> operator+(CoDouble<double> a,double b)
	{
		CoDouble<double> r = a;
		return r += b;
	}

	CoDouble<double> operator+(double a,CoDouble<double> b)
	{
		CoDouble<double> r = b;
		return r += a;
	}

	CoDouble<double> operator*(CoDouble<double> a,CoDouble<double> b)
	{
		CoDouble<double> r = a;
		return r *= b;
	}

	CoDouble<double> operator*(CoDouble<double> a,double b)
	{
		CoDouble<double> r = a;
		return r *= b;
	}

	CoDouble<double> operator*(double a,CoDouble<double> b)
	{
		CoDouble<double> r = b;
		return r *= a;
	}

	CoDouble<double> operator-(CoDouble<double> a,CoDouble<double> b)
	{
		CoDouble<double> r = a;
		return r -= b;
	}

	CoDouble<double> operator-(CoDouble<double> a,double b)
	{
		CoDouble<double> r = a;
		return r -= b;
	}

	CoDouble<double> operator-(double a,CoDouble<double> b)
	{
		CoDouble<double> r = b;
		return r -= a;
	}

	CoDouble<double> operator/(CoDouble<double> a,CoDouble<double> b)
	{
		CoDouble<double> r = a;
		return r /= b;
	}

	CoDouble<double> operator/(CoDouble<double> a,double b)
	{
		CoDouble<double> r = a;
		return r /= b;
	}

	CoDouble<double> operator/(double a,CoDouble<double> b)
	{
		CoDouble<double> r = b;
		return r /= a;
	}

	bool operator==(CoDouble<double> a,CoDouble<double> b)
	{
		return a.val() == b.val();
	}

	bool operator!=(CoDouble<double> a,CoDouble<double> b)
	{
		return a.val() != b.val();
	}

	CoFloat<float> operator+(CoFloat<float> a,CoFloat<float> b)
	{
		CoFloat<float> r = a;
		return r += b;
	}

	CoFloat<float> operator+(CoFloat<float> a,float b)
	{
		CoFloat<float> r = a;
		return r += b;
	}

	CoFloat<float> operator+(float a,CoFloat<float> b)
	{
		CoFloat<float> r = b;
		return r += a;
	}

	CoFloat<float> operator*(CoFloat<float> a,CoFloat<float> b)
	{
		CoFloat<float> r = a;
		return r *= b;
	}

	CoFloat<float> operator*(CoFloat<float> a,float b)
	{
		CoFloat<float> r = a;
		return r *= b;
	}

	CoFloat<float> operator*(float a,CoFloat<float> b)
	{
		CoFloat<float> r = b;
		return r *= a;
	}

	CoFloat<float> operator-(CoFloat<float> a,CoFloat<float> b)
	{
		CoFloat<float> r = a;
		return r -= b;
	}

	CoFloat<float> operator-(CoFloat<float> a,float b)
	{
		CoFloat<float> r = a;
		return r -= b;
	}

	CoFloat<float> operator-(float a,CoFloat<float> b)
	{
		CoFloat<float> r = b;
		return r -= a;
	}

	CoFloat<float> operator/(CoFloat<float> a,CoFloat<float> b)
	{
		CoFloat<float> r = a;
		return r /= b;
	}

	CoFloat<float> operator/(CoFloat<float> a,float b)
	{
		CoFloat<float> r = a;
		return r /= b;
	}

	CoFloat<float> operator/(float a,CoFloat<float> b)
	{
		CoFloat<float> r = b;
		return r /= a;
	}

	bool operator==(CoFloat<float> a,CoFloat<float> b)
	{
		return a.val() == b.val();
	}

	bool operator!=(CoFloat<float> a,CoFloat<float> b)
	{
		return a.val() != b.val();
	}

	CoUInt<unsigned> operator+(CoUInt<unsigned> a,CoUInt<unsigned> b)
	{
		CoUInt<unsigned> r = a;
		return r += b;
	}

	CoUInt<unsigned> operator+(CoUInt<unsigned> a,unsigned b)
	{
		CoUInt<unsigned> r = a;
		return r += b;
	}

	CoUInt<unsigned> operator+(unsigned a,CoUInt<unsigned> b)
	{
		CoUInt<unsigned> r = b;
		return r += a;
	}

	CoUInt<unsigned> operator*(CoUInt<unsigned> a,CoUInt<unsigned> b)
	{
		CoUInt<unsigned> r = a;
		return r *= b;
	}

	CoUInt<unsigned> operator*(CoUInt<unsigned> a,unsigned b)
	{
		CoUInt<unsigned> r = a;
		return r *= b;
	}

	CoUInt<unsigned> operator*(unsigned a,CoUInt<unsigned> b)
	{
		CoUInt<unsigned> r = b;
		return r *= a;
	}

	CoUInt<unsigned> operator-(CoUInt<unsigned> a,CoUInt<unsigned> b)
	{
		CoUInt<unsigned> r = a;
		return r -= b;
	}

	CoUInt<unsigned> operator-(CoUInt<unsigned> a,unsigned b)
	{
		CoUInt<unsigned> r = a;
		return r -= b;
	}

	CoUInt<unsigned> operator-(unsigned a,CoUInt<unsigned> b)
	{
		CoUInt<unsigned> r = b;
		return r -= a;
	}

	CoUInt<unsigned> operator/(CoUInt<unsigned> a,CoUInt<unsigned> b)
	{
		CoUInt<unsigned> r = a;
		return r /= b;
	}

	CoUInt<unsigned> operator/(CoUInt<unsigned> a,unsigned b)
	{
		CoUInt<unsigned> r = a;
		return r /= b;
	}

	CoUInt<unsigned> operator/(unsigned a,CoUInt<unsigned> b)
	{
		CoUInt<unsigned> r = b;
		return r /= a;
	}

	bool operator==(CoUInt<unsigned> a,CoUInt<unsigned> b)
	{
		return a.val() == b.val();
	}

	bool operator!=(CoUInt<unsigned> a,CoUInt<unsigned> b)
	{
		return a.val() != b.val();
	}

	CoBool<bool> operator+(CoBool<bool> a,CoBool<bool> b)
	{
		CoBool<bool> r = a;
		return r += b;
	}

	CoBool<bool> operator+(CoBool<bool> a,bool b)
	{
		CoBool<bool> r = a;
		return r += b;
	}

	CoBool<bool> operator+(bool a,CoBool<bool> b)
	{
		CoBool<bool> r = b;
		return r += a;
	}

	CoBool<bool> operator*(CoBool<bool> a,CoBool<bool> b)
	{
		CoBool<bool> r = a;
		return r *= b;
	}

	CoBool<bool> operator*(CoBool<bool> a,bool b)
	{
		CoBool<bool> r = a;
		return r *= b;
	}

	CoBool<bool> operator*(bool a,CoBool<bool> b)
	{
		CoBool<bool> r = b;
		return r *= a;
	}

	CoBool<bool> operator-(CoBool<bool> a,CoBool<bool> b)
	{
		CoBool<bool> r = a;
		return r -= b;
	}

	CoBool<bool> operator-(CoBool<bool> a,bool b)
	{
		CoBool<bool> r = a;
		return r -= b;
	}

	CoBool<bool> operator-(bool a,CoBool<bool> b)
	{
		CoBool<bool> r = b;
		return r -= a;
	}

	CoBool<bool> operator/(CoBool<bool> a,CoBool<bool> b)
	{
		CoBool<bool> r = a;
		return r /= b;
	}

	CoBool<bool> operator/(CoBool<bool> a,bool b)
	{
		CoBool<bool> r = a;
		return r /= b;
	}

	CoBool<bool> operator/(bool a,CoBool<bool> b)
	{
		CoBool<bool> r = b;
		return r /= a;
	}

	bool operator==(CoBool<bool> a,CoBool<bool> b)
	{
		return a.val() == b.val();
	}

	bool operator!=(CoBool<bool> a,CoBool<bool> b)
	{
		return a.val() != b.val();
	}

	//create nonmember arithmetic,equality, and IO operator overloads for complex data types
	CoCDouble<cufftDoubleComplex> operator+(CoCDouble<cufftDoubleComplex> a,CoCDouble<cufftDoubleComplex> b)
	{
		CoCDouble<cufftDoubleComplex> r = a;
		return r += b;
	}

	CoCDouble<cufftDoubleComplex> operator+(CoCDouble<cufftDoubleComplex> a,cufftDoubleComplex b)
	{
		CoCDouble<cufftDoubleComplex> r = a;
		return r += b;
	}

	CoCDouble<cufftDoubleComplex> operator+(cufftDoubleComplex a,CoCDouble<cufftDoubleComplex> b)
	{
		CoCDouble<cufftDoubleComplex> r = b;
		return r += a;
	}

	CoCDouble<cufftDoubleComplex> operator*(CoCDouble<cufftDoubleComplex> a,CoCDouble<cufftDoubleComplex> b)
	{
		CoCDouble<cufftDoubleComplex> r = a;
		return r *= b;
	}

	CoCDouble<cufftDoubleComplex> operator*(CoCDouble<cufftDoubleComplex> a,cufftDoubleComplex b)
	{
		CoCDouble<cufftDoubleComplex> r = a;
		return r *= b;
	}

	CoCDouble<cufftDoubleComplex> operator*(cufftDoubleComplex a,CoCDouble<cufftDoubleComplex> b)
	{
		CoCDouble<cufftDoubleComplex> r = b;
		return r *= a;
	}

	CoCDouble<cufftDoubleComplex> operator-(CoCDouble<cufftDoubleComplex> a,CoCDouble<cufftDoubleComplex> b)
	{
		CoCDouble<cufftDoubleComplex> r = a;
		return r -= b;
	}

	CoCDouble<cufftDoubleComplex> operator-(CoCDouble<cufftDoubleComplex> a,cufftDoubleComplex b)
	{
		CoCDouble<cufftDoubleComplex> r = a;
		return r -= b;
	}

	CoCDouble<cufftDoubleComplex> operator-(cufftDoubleComplex a,CoCDouble<cufftDoubleComplex> b)
	{
		CoCDouble<cufftDoubleComplex> r = b;
		return r -= a;
	}

	CoCDouble<cufftDoubleComplex> operator/(CoCDouble<cufftDoubleComplex> a,CoCDouble<cufftDoubleComplex> b)
	{
		CoCDouble<cufftDoubleComplex> r = a;
		return r /= b;
	}

	CoCDouble<cufftDoubleComplex> operator/(CoCDouble<cufftDoubleComplex> a,cufftDoubleComplex b)
	{
		CoCDouble<cufftDoubleComplex> r = a;
		return r /= b;
	}

	CoCDouble<cufftDoubleComplex> operator/(cufftDoubleComplex a,CoCDouble<cufftDoubleComplex> b)
	{
		CoCDouble<cufftDoubleComplex> r = b;
		return r /= a;
	}

	bool operator==(CoCDouble<cufftDoubleComplex> a,CoCDouble<cufftDoubleComplex> b)
	{
		return a.val().x == b.val().x && a.val().y == b.val().y;
	}

	bool operator!=(CoCDouble<cufftDoubleComplex> a,CoCDouble<cufftDoubleComplex> b)
	{
		return a.val().x != b.val().x || a.val().y != b.val().y;
	}

	CoCFloat<cufftComplex> operator+(CoCFloat<cufftComplex> a,CoCFloat<cufftComplex> b)
	{
		CoCFloat<cufftComplex> r = a;
		return r += b;
	}

	CoCFloat<cufftComplex> operator+(CoCFloat<cufftComplex> a,cufftComplex b)
	{
		CoCFloat<cufftComplex> r = a;
		return r += b;
	}

	CoCFloat<cufftComplex> operator+(cufftComplex a,CoCFloat<cufftComplex> b)
	{
		CoCFloat<cufftComplex> r = b;
		return r += a;
	}

	CoCFloat<cufftComplex> operator*(CoCFloat<cufftComplex> a,CoCFloat<cufftComplex> b)
	{
		CoCFloat<cufftComplex> r = a;
		return r *= b;
	}

	CoCFloat<cufftComplex> operator*(CoCFloat<cufftComplex> a,cufftComplex b)
	{
		CoCFloat<cufftComplex> r = a;
		return r *= b;
	}

	CoCFloat<cufftComplex> operator*(cufftComplex a,CoCFloat<cufftComplex> b)
	{
		CoCFloat<cufftComplex> r = b;
		return r *= a;
	}

	CoCFloat<cufftComplex> operator-(CoCFloat<cufftComplex> a,CoCFloat<cufftComplex> b)
	{
		CoCFloat<cufftComplex> r = a;
		return r -= b;
	}

	CoCFloat<cufftComplex> operator-(CoCFloat<cufftComplex> a,cufftComplex b)
	{
		CoCFloat<cufftComplex> r = a;
		return r -= b;
	}

	CoCFloat<cufftComplex> operator-(cufftComplex a,CoCFloat<cufftComplex> b)
	{
		CoCFloat<cufftComplex> r = b;
		return r -= a;
	}

	CoCFloat<cufftComplex> operator/(CoCFloat<cufftComplex> a,CoCFloat<cufftComplex> b)
	{
		CoCFloat<cufftComplex> r = a;
		return r /= b;
	}

	CoCFloat<cufftComplex> operator/(CoCFloat<cufftComplex> a,cufftComplex b)
	{
		CoCFloat<cufftComplex> r = a;
		return r /= b;
	}

	CoCFloat<cufftComplex> operator/(cufftComplex a,CoCFloat<cufftComplex> b)
	{
		CoCFloat<cufftComplex> r = b;
		return r /= a;
	}

	bool operator==(CoCFloat<cufftComplex> a,CoCFloat<cufftComplex> b)
	{
		return a.val().x == b.val().x && a.val().y == b.val().y;
	}

	bool operator!=(CoCFloat<cufftComplex> a,CoCFloat<cufftComplex> b)
	{
		return a.val().x != b.val().x || a.val().y != b.val().y;
	}

	//Complex utility functions
	//float
	CoCFloat<cufftComplex> polar(float rho, float theta)
	{
		cufftComplex c;
		c.x = rho*std::cos(theta);
		c.y = rho*std::sin(theta);
		return CoCFloat<cufftComplex>(c);
	}
	CoCFloat<cufftComplex> conj(CoCFloat<cufftComplex> c)
	{
		cufftComplex b;
		b.x = c.val().x;
		b.y = -c.val().y;
		return CoCFloat(b);
	}

	float abs(CoCFloat<cufftComplex> c)
	{
		return std::sqrt(real(c)*real(c) + imag(c)*imag(c));
	}

	float arg(CoCFloat<cufftComplex> c)
	{
		return std::arctan(c.val().y / c.val().x);
	}

	float norm(CoCFloat<cufftComplex> c)
	{
		return std::sqrt(real(c)*real(c) + imag(c)*imag(c));
	}

	float real(CoCFloat<cufftComplex> c)
	{
		return c.val().x;
	}

	float imag(CoCFloat<cufftComplex> c)
	{
		return c.val().y;
	}

	//double
	CoCDouble<cufftDoubleComplex> polar(double rho, double theta)
	{
		cufftDoubleComplex c;
		c.x = rho*std::cos(theta);
		c.y = rho*std::sin(theta);
		return CoCDouble<cufftDoubleComplex>(c);
	}
	CoCDouble<cufftDoubleComplex> conj(CoCDouble<cufftDoubleComplex> c)
	{
		cufftDoubleComplex b;
		b.x = c.val().x;
		b.y = -c.val().y;
		return CoCDouble(b);
	}

	double abs(CoCDouble<cufftDoubleComplex> c)
	{
		return std::sqrt(real(c)*real(c) + imag(c)*imag(c));
	}

	double arg(CoCDouble<cufftDoubleComplex> c)
	{
		return std::arctan(c.val().y / c.val().x);
	}

	double norm(CoCDouble<cufftDoubleComplex> c)
	{
		return std::sqrt(real(c)*real(c) + imag(c)*imag(c));
	}

	double real(CoCDouble<cufftDoubleComplex> c)
	{
		return c.val().x;
	}

	double imag(CoCDouble<cufftDoubleComplex> c)
	{
		return c.val().y;
	}
}
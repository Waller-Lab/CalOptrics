//	co_exceptions.h
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

#ifndef CO_EXCEPTIONS_H
#define CO_EXCEPTIONS_H

namespace co 
{
	class COerr {
		virtual void printError(const char *msg) const;
	};
	//Host Errors
	class HostErr : public COerr {
		virtual void printError(const char *msg) const;
	};
	class InvalidArgErr : public HostErr {
		virtual void printError(const char *msg) const;
	};
	//Device Errors
	//Cuda Errors
	class CudaErr : COerr {
		virtual void printError(const char *msg) const;
	};
	class CudaSetDeviceErr : public CudaErr {
		virtual void printError(const char *msg) const;
	};
	class CudaDevSyncErr : public CudaErr {
		virtual void printError(const char *msg) const;
	};
	class CudaMallocErr : public CudaErr {
		virtual void printError(const char *msg) const;
	};
	class CudaMemcpyErr : public CudaErr {
		virtual void printError(const char *msg) const;
	};
	class CudaFFTErr : public CudaErr {
		virtual void printError(const char *msg) const;
	};
	class CudaBLASErr : public CudaErr {
		virtual void printError(const char *msg) const;
	};
	class CudaThrustErr : public CudaErr {
		virtual void printError(const char *msg) const;
	};
}

#endif
//	co_exceptions.cpp
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

#include <iostream>
#include "co_exceptions.h"

namespace co {
	
	//COerr member functions
	void COerr::printError(const char *msg) const 
	{
		std::cerr << "Unknown CalOptrics error occurred: " << msg << std::endl;
	}
	
	//HostErr member functions
	void HostErr::printError(const char *msg) const 
	{
		std::cerr << "Unknown Caloptrics Host Error: " << msg << std::endl;
	}

	//InvalidArgErr member functions
	void InvalidArgErr::printError(const char *msg) const 
	{
		std::cerr << "Invalid Argument Error: " << msg << std::endl;
	}

	//CudaErr member functions
	void CudaErr::printError(const char *msg) const
	{
		std::cerr << "Cuda Error occurred: " << msg << std::endl;
	}

	//CudaSetDeviceErr member functions
	void CudaSetDeviceErr::printError(const char *msg) const
	{
		std::cerr << "Cuda Error occurred: " << msg << std::endl;
	}

	//CudaDevSynstd::cerr member functions
	void CudaDevSyncErr::printError(const char *msg) const 
	{
		std::cerr << "Cuda Device Sync Error occurred: " << msg << std::endl;
	}

	//CudaMallostd::cerr member functions
	void CudaMallocErr::printError(const char *msg) const 
	{
		std::cerr << "Cuda Malloc Error occurred: " << msg << std::endl;
	}

	//CudaMemcpyErr member functions
	void CudaMemcpyErr::printError(const char *msg) const 
	{
		std::cerr << "Cuda Memcpy Error occurred: " << msg << std::endl;
	}

	//CudaFFTErr member functions
	void CudaFFTErr::printError(const char *msg) const 
	{
		std::cerr << "Cuda FFT Error occurred: " << msg << std::endl;
	}

	//CudaBLASErr member functions
	void CudaBLASErr::printError(const char *msg) const 
	{
		std::cerr << "Cuda BLAS Error occurred: " << msg << std::endl;
	}

	//CudaThrustErr member functions
	void CudaThrustErr::printError(const char *msg) const 
	{
		std::cerr << "Cuda Thrust Error occurred: " << msg << std::endl;
	}
}
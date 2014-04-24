#include <thrust/version.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>
#include <list>
#include <vector>

#include "C:\Users\Diiv\git\CalOptrics\Cpp_library_windows\CalOptrics\CalOptrics\caloptrics.h"

void thrustVersionAndVectorExamples(); 
void thrustCopyFillSequenceExamples();
void thrustVectorListExample();
void anotherThrustTest();
void quitProgramPrompt(bool);
void coDataTypeSanityChecks();

using namespace co;

int main(void)
{
    //thrustVersionAndVectorExamples();
	//thrustCopyFillSequenceExamples();
	//thrustVectorListExample();
	//anotherThrustTest();
	coDataTypeSanityChecks();
	

	quitProgramPrompt(true);
    return 0;
}

void coDataTypeSanityChecks()
{
	Float f1 = Float(3.14);
	Float f2 = Float(2.14);
	std::cout << (f1.val()==3.14f) << std::endl;
	std::cout << ((f1*f1-3.14f*3.14f).val() < .0001) << std::endl;
	std::cout << (((f1+f2)-(3.14f+2.14f)).val() < .0001) << std::endl;

	Double d1 = Double(3.14);
	Double d2 = Double(2.14);
	std::cout << (d1.val()==3.14) << std::endl;
	std::cout << (((d1*d1-3.14*3.14)).val() < .0001) << std::endl;
	std::cout << (((d1+d2)-(3.14+2.14)).val() < .0001) << std::endl;
	

	Bool b1 = Bool(true);
	Bool b2 = Bool(false);
	std::cout << (b1.val() == true) << std::endl;
	std::cout << (b2.val() == false) << std::endl;
	std::cout << (b1 != b2) << std::endl;
	std::cout << (!(b1 == b2)) << std::endl;
	std::cout << (!(b1 && b2)).val() << std::endl;

	Int i1 = Int(4);
	Int i2 = Int(9);
	std::cout << (i1.val()==4) << std::endl;
	std::cout << ((i1*i1).val() == 16) << std::endl;
	std::cout << ((i1+i2).val() == 13) << std::endl;

	cufftComplex c1;
	c1.x = 2;
	c1.y = 3;
	CFloat cf1 = CFloat(c1);

	cufftComplex c2;
	c2.x = 2;
	c2.y = 2;
	CFloat cf2 = CFloat(c2);

	std::cout << cf1.val().x << " " << cf1.val().y << std::endl;
	std::cout << cf2.val().x << " " << cf2.val().y << std::endl;
	std::cout << (cf1*cf2).val().x << " " << (cf1*cf2).val().y << std::endl;
	std::cout << (cf2/cf2).val().x << " " << (cf2/cf2).val().y << std::endl;
}

void anotherThrustTest()
{
	thrust::device_vector<int>* D;
	thrust::host_vector<int>* H;
	D = new thrust::device_vector<int>(3,1);
	H = new thrust::host_vector<int>(*D);

	std::vector<int>* nums = new std::vector<int>;
	nums->push_back(1);
	nums->push_back(1);
	nums->push_back(1);
	thrust::host_vector<int>* N = new thrust::host_vector<int>(*nums);


	for(int i = 0;i < N->size(); i++){
		std::cout << "N[" << i << "]= " << (*N)[i] << std::endl; 
	}
}

void thrustVectorListExample()
{
	// create an STL list with 4 values
	std::list<int> stl_list;

	stl_list.push_back(10);
	stl_list.push_back(20);
	stl_list.push_back(30);
	stl_list.push_back(40);

	// initialize a device_vector with the list
	thrust::device_vector<int> D(stl_list.begin(), stl_list.end());

	// print D
    for(int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

	std::cout << std::endl;

	// copy a device_vector into an STL vector
	std::vector<int> stl_vector(D.size());
	thrust::copy(D.begin(), D.end(), stl_vector.begin());

	// print stl_vector
    for(int i = 0; i < stl_vector.size(); i++)
        std::cout << "stl_vector[" << i << "] = " << stl_vector[i] << std::endl;
}

void thrustCopyFillSequenceExamples()
{
	 // initialize all ten integers of a device_vector to 1
    thrust::device_vector<int> D(10, 1);

    // set the first seven elements of a vector to 9
    thrust::fill(D.begin(), D.begin() + 7, 9);

    // initialize a host_vector with the first five elements of D
    thrust::host_vector<int> H(D.begin(), D.begin() + 5);

    // set the elements of H to 0, 1, 2, 3, ...
    thrust::sequence(H.begin(), H.end());

    // copy all of H back to the beginning of D
    thrust::copy(H.begin(), H.end(), D.begin());

    // print D
    for(int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;
}


void thrustVersionAndVectorExamples()
{
	int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;

    std::cout << "Thrust v" << major << "." << minor << std::endl;

	// H has storage for 4 integers
	thrust::host_vector<int> H(4);

	//initialize individual elements
	H[0] = 14;
	H[1] = 20;
	H[2] = 38;
	H[3] = 46;

	// H.size() returns the size of vector H
	std::cout << "H has size " << H.size() << std::endl;

	//print contents of H
	for(unsigned i = 0;i < H.size(); i++)
		std::cout << "H[" << i << "] = " << H[i] << std::endl;

	// resize H
	H.resize(2);

	std::cout << "After resize, H now has size " << H.size() << std::endl;

	//print contents of H
	for(unsigned i = 0;i < H.size(); i++)
		std::cout << "H[" << i << "] = " << H[i] << std::endl;

	// Copy host_vector H to device_vector D
	thrust::device_vector<int> D = H;

	// elements of D can be modfied
	D[0] = 99;
	D[1] = 88;
	
	// print contents of D
	for(unsigned i = 0;i < D.size(); i++)
		std::cout << "D[" << i << "] = " << D[i] << std::endl;

	// print contents of H
	for(unsigned i = 0;i < H.size(); i++)
		std::cout << "H[" << i << "] = " << H[i] << std::endl;
	
	// H and D are automatically deleted when the function returns
}


/* host functions */
void quitProgramPrompt(bool success)
{
  int c;
  if(success)
	printf( "\nProgram Executed Successfully. Press ENTER to quit program...\n" );
  else
	printf( "\nProgram Execution Failed. Press ENTER to quit program...\n" );
  fflush( stdout );
  do c = getchar(); while ((c != '\n') && (c != EOF));
}
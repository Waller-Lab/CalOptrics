#include <thrust/version.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>
#include <list>
#include <vector>

void thrustVersionAndVectorExamples(); 
void thrustCopyFillSequenceExamples();
void thrustVectorListExample();
void quitProgramPrompt(bool);

int main(void)
{
    //thrustVersionAndVectorExamples();
	//thrustCopyFillSequenceExamples();
	thrustVectorListExample();
	

	quitProgramPrompt(true);
    return 0;
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
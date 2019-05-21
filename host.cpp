#include <iostream>
#include <tuple>
#include <assert.h>

#include "SimpleCL.h"

using namespace std;
 
int main()
{
	// kernel calculates for each element C=A+B
	std::string kernel_code=
			"void kernel simple_add(global const int* A, global const int* B, global int* C)"
			"{"
			"	C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];"
			"}";

	SimpleCLContext context(kernel_code, "simple_add");

	int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
  
	// create buffers on the device
	cl::Buffer buffer_A = context.createInitBuffer(sizeof(int)*10, A, true);
	cl::Buffer buffer_B = context.createInitBuffer(sizeof(int)*10, B, true);
	cl::Buffer buffer_C = context.createBuffer(sizeof(int)*10);

	context.setArgs(buffer_A, buffer_B, buffer_C);
 
	assert(CL_SUCCESS == context.queue.enqueueNDRangeKernel(context.kernel,cl::NullRange,cl::NDRange(10),cl::NullRange));
	assert(CL_SUCCESS == context.queue.finish());
 
	int C[10];
	//read result C from the device to array C
	context.queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*10,C);
 
	std::cout<<" result: \n";
	for(int i=0;i<10;i++){
		std::cout<<C[i]<<" ";
	}
 
	return 0;
}

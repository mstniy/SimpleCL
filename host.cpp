#include <iostream>
#include <assert.h>

#include "SimpleCL.h"

using namespace std;
 
int main()
{
	int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
 	int C[10];

	assert(SimpleCLContext().isNull());

	SimpleCLContext context("host.cl");

	assert(context.isNull() == false);
  
	cl::Buffer buffer_A = context.createInitBuffer(sizeof(int)*10, A, SimpleCLReadOnly);
	cl::Buffer buffer_B = context.createInitBuffer(sizeof(int)*10, B, SimpleCLReadOnly);
	cl::Buffer buffer_C = context.createBuffer(sizeof(int)*10);

	SimpleCLKernel addKernel(context.createKernel("simple_add"));
	cout << "addKernel max workgroupsize: " << addKernel.getMaxWorkGroupSize() << endl;

	addKernel(cl::NDRange(10), buffer_A, buffer_B, buffer_C);

	context.readBuffer(C, buffer_C, sizeof(int)*10);
 
	cout << " result: " << endl;
	for(int i=0;i<10;i++)
		cout << C[i] << " ";
	cout << endl;

	SimpleCLKernel sumKernel(context.createKernel("sum_reduce"));
	size_t sumKernelMWGS = sumKernel.getMaxWorkGroupSize();
	cout << "sumKernel max workgroupsize: " << sumKernelMWGS << endl;

	sumKernel(cl::NDRange(10), buffer_C, SimpleCLLocalMemory<cl_float>(sumKernelMWGS), 10);

	context.readBuffer(C, buffer_C, sizeof(int)*10);
 
	cout << " result: " << endl;
	for(int i=0;i<10;i++)
		cout << C[i] << " ";
	cout << endl;
 
	return 0;
}

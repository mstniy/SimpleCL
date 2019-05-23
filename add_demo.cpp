#include <iostream>
#include <assert.h>
#include <memory>

#include "SimpleCL.h"

using namespace std;
 
int main()
{
	const size_t N = 1000000;
	const int M = 3;
	unique_ptr<cl_float[]> A(new cl_float[N]);
	unique_ptr<cl_float[]> B(new cl_float[N]);
	unique_ptr<cl_float[]> C(new cl_float[N]);
	for (int i=0; i<N; i++)
	{
		A[i] = i;
		B[i] = M-i;
	}

	assert(SimpleCLContext().isNull());

	SimpleCLContext context("../add_demo.cl");

	assert(context.isNull() == false);
  
	cl::Buffer buffer_A = context.createInitBuffer(sizeof(cl_float)*N, A.get(), SimpleCLReadOnly);
	cl::Buffer buffer_B = context.createInitBuffer(sizeof(cl_float)*N, B.get(), SimpleCLReadOnly);
	cl::Buffer buffer_C = context.createBuffer(sizeof(cl_float)*N);

	SimpleCLKernel addKernel(context.createKernel("simple_add"));

	addKernel(cl::NDRange(N), cl::NullRange, buffer_A, buffer_B, buffer_C);

	context.readBuffer(C.get(), buffer_C, sizeof(cl_float)*N);
 
	cout << " result: " << endl;
	for(int i=0; i<std::min((size_t)10, N); i++)
		cout << C[i] << " ";
	cout << "..." << endl;

	for(int i=0; i<N; i++)
		assert(C[i] == M);

	cout << "Correct answer" << endl;

	return 0;
}

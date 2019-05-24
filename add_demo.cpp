#include <iostream>
#include <assert.h>
#include <math.h>
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
	for (size_t i=0; i<N; i++)
	{
		A[i] = i;
		B[i] = M-int(i);
	}

	assert(SimpleCLContext().isNull());

	SimpleCLContext context("../add_demo.cl");

	assert(context.isNull() == false);
  
	SimpleCLBuffer<cl_float> buffer_A = context.createInitBuffer<cl_float>(N, A.get(), SimpleCLReadOnly);
	SimpleCLBuffer<cl_float> buffer_B = context.createInitBuffer<cl_float>(N, B.get(), SimpleCLReadOnly);
	SimpleCLBuffer<cl_float> buffer_C = context.createBuffer<cl_float>(N);

	SimpleCLKernel addKernel(context.createKernel("simple_add"));

	addKernel(cl::NDRange(N), cl::NullRange, buffer_A, buffer_B, buffer_C);

	buffer_C.read(C.get(), N);
 
	cout << " result: " << endl;
	for(size_t i=0; i<std::min((size_t)10, N); i++)
		cout << C[i] << " ";
	cout << "..." << endl;

	for(size_t i=0; i<N; i++)
		assert(fabs(C[i]-M)<1e-6);

	cout << "Correct answer" << endl;

	return 0;
}

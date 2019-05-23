#include <iostream>
#include <assert.h>
#include <memory>

#include "SimpleCL.h"

using namespace std;

size_t decideWorkGroupSize(size_t N, size_t mwgs)
{
	/*if (N < mwgs)
		return N;*/
	for (int i=mwgs; i>1; i--) // This is assuming that mwgs is going to be small. On my system its 1024.
		if (N%i == 0)
			return i;
	return 1; // Prime?
}
 
int main()
{
	const size_t N = 1000000; // Be careful to not choose a prime(ish) N! Workgroup size must be divide N, and sum_reduce kernel produces as many sums as there are workgroups. So if N is prime, workgroup size will be 1 (if N > maximumworkgroupsize) and sum_reduce will basically not do anything and the CPU will do the addition.
	const int M = 3;
	unique_ptr<cl_float[]> A(new cl_float[N]);
	unique_ptr<cl_float[]> B(new cl_float[N]);
	unique_ptr<cl_float[]> C(new cl_float[N]);
	for (int i=0; i<N; i++)
	{
		A[i] = i;
		B[i] = 3-i;
	}

	assert(SimpleCLContext().isNull());

	SimpleCLContext context("host.cl");

	assert(context.isNull() == false);
  
	cl::Buffer buffer_A = context.createInitBuffer(sizeof(cl_float)*N, A.get(), SimpleCLReadOnly);
	cl::Buffer buffer_B = context.createInitBuffer(sizeof(cl_float)*N, B.get(), SimpleCLReadOnly);
	cl::Buffer buffer_C = context.createBuffer(sizeof(cl_float)*N);

	SimpleCLKernel addKernel(context.createKernel("simple_add"));
	cout << "addKernel max workgroupsize: " << addKernel.getMaxWorkGroupSize() << endl;

	addKernel(cl::NDRange(N), cl::NullRange, buffer_A, buffer_B, buffer_C);

	context.readBuffer(C.get(), buffer_C, sizeof(cl_float)*N);
 
	cout << " result: " << endl;
	for(int i=0;i<std::min((size_t)10, N);i++)
		cout << C[i] << " ";
	cout << "..." << endl;

	SimpleCLKernel sumKernel(context.createKernel("sum_reduce"));
	size_t sumKernelWGS = decideWorkGroupSize(N, sumKernel.getMaxWorkGroupSize());
	assert(N % sumKernelWGS == 0);
	cout << "sumKernel workgroup size: " << sumKernelWGS << endl;
	size_t nowg = N/sumKernelWGS;

	unique_ptr<cl_float[]> sum_output(new cl_float[nowg]);
	cl::Buffer buffer_sum_output = context.createBuffer(sizeof(cl_float)*nowg);

	sumKernel(cl::NDRange(N), cl::NDRange(sumKernelWGS), buffer_C, SimpleCLLocalMemory<cl_float>(sumKernelWGS), buffer_sum_output);

	context.readBuffer(sum_output.get(), buffer_sum_output, sizeof(cl_float)*nowg);
 
	cout << " result: " << endl;
	for(int i=0;i<std::min((size_t)10,nowg);i++)
		cout << sum_output[i] << " ";
	cout << "..." << endl;

	int bigSum=0;
	for (int i=0; i<nowg; i++)
		bigSum += sum_output[i];

	assert(bigSum == 3*N);
	cout << "Correct answer" << endl;
 
	return 0;
}

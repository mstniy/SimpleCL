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
	for (size_t i=0; i<N; i++)
		A[i] = M;

	assert(SimpleCLContext().isNull());

	SimpleCLContext context("../sum_demo.cl");

	assert(context.isNull() == false);
  
	SimpleCLBuffer<cl_float> buffer_A = context.createInitBuffer<cl_float>(N, A.get(), SimpleCLReadOnly);

	SimpleCLKernel sumKernel(context.createKernel("sum_reduce"));
	size_t sumKernelWGS = std::min(N, sumKernel.getMaxWorkGroupSize());
	cout << "sumKernel workgroup size: " << sumKernelWGS << endl;
	int Nrounded = N%sumKernelWGS == 0 ? N : N+sumKernelWGS-(N%sumKernelWGS); // We might create at most maxworkgroupsize-1 many unnecessary threads (Basically one more workgroup).
	size_t nowg = Nrounded/sumKernelWGS;

	unique_ptr<cl_float[]> sum_output(new cl_float[nowg]);
	SimpleCLBuffer<cl_float> buffer_sum_output = context.createBuffer<cl_float>(nowg);

	sumKernel(cl::NDRange(Nrounded), cl::NDRange(sumKernelWGS), buffer_A, SimpleCLLocalMemory<cl_float>(sumKernelWGS), buffer_sum_output, (cl_int)N);

	buffer_sum_output.read(sum_output.get(), nowg);
 
	cout << " result: " << endl;
	for(size_t i=0;i<std::min((size_t)10, nowg);i++)
		cout << sum_output[i] << " ";
	cout << "..." << endl;

	int bigSum=0;
	for (size_t i=0; i<nowg; i++)
		bigSum += sum_output[i];

	cout << "Expected " << (3*N) << " got " << bigSum << endl;
	assert(bigSum == 3*N);
	cout << "Correct answer" << endl;
 
	return 0;
}

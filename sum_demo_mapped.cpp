#include <iostream>
#include <assert.h>
#include <memory>
#include <math.h>

#include "SimpleCL.h"

using namespace std;
 
int main()
{
	const size_t N = 1000000;
	const int M = 3;

	assert(SimpleCLContext().isNull());

	SimpleCLContext context("../sum_demo.cl");

	assert(context.isNull() == false);
  
	SimpleCLBuffer<cl_float> buffer_A = context.createBuffer<cl_float>(N, SimpleCLRead|SimpleCLHostAlloc);

	volatile cl_float* A = buffer_A.map();

	for (size_t i=0; i<N; i++)
		A[i] = M;

	buffer_A.unmap(A);

	SimpleCLKernel sumKernel(context.createKernel("sum_reduce"));
	size_t sumKernelWGS = std::min(N, sumKernel.getMaxWorkGroupSize());
	cout << "sumKernel workgroup size: " << sumKernelWGS << endl;
	int Nrounded = N%sumKernelWGS == 0 ? N : N+sumKernelWGS-(N%sumKernelWGS); // We might create at most maxworkgroupsize-1 many unnecessary threads (Basically one more workgroup).
	size_t nowg = Nrounded/sumKernelWGS;

	SimpleCLBuffer<cl_float> buffer_sum_output = context.createBuffer<cl_float>(nowg, SimpleCLReadWrite|SimpleCLHostAlloc);

	sumKernel(cl::NDRange(Nrounded), cl::NDRange(sumKernelWGS), buffer_A, SimpleCLLocalMemory<cl_float>(sumKernelWGS), buffer_sum_output, (cl_int)N);

	volatile cl_float* sum_output = buffer_sum_output.map();
 
	cout << " result: " << endl;
	for(size_t i=0;i<std::min((size_t)10, nowg);i++)
		cout << sum_output[i] << " ";
	cout << "..." << endl;

	int bigSum=0;
	for (size_t i=0; i<nowg; i++)
		bigSum += sum_output[i];

	cout << "Expected " << (M*N) << " got " << bigSum << endl;
	assert(fabs(bigSum-M*N) < 1e-6);
	cout << "Correct answer" << endl;
 
	return 0;
}

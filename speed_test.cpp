#include <iostream>
#include <assert.h>
#include <memory>
#include <math.h>
#include <chrono>
#include <unistd.h>

#include "SimpleCL.h"

using namespace std;

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

const int CYCLE_COUNT = 1000;
 
uint64_t bench_copy(size_t N)
{
	unique_ptr<cl_float[]> arr(new cl_float[N]);
	unique_ptr<cl_float[]> output(new cl_float[N]);

	SimpleCLContext context("../speed_test.cl");

	SimpleCLKernel kernel(context.createKernel("increment"));
	SimpleCLBuffer<cl_float> buffer_A = context.createBuffer<cl_float>(N, SimpleCLRead); // In a real application, we can alloate the buffers once and reuse them over and over again. So the buffer allocation aren't included in the measure the here.
	SimpleCLBuffer<cl_float> buffer_output = context.createBuffer<cl_float>(N, SimpleCLWrite);

	TimePoint start, end;

	for (int cycle=0; cycle<CYCLE_COUNT; cycle++)
	{
		for (size_t i=0; i<N; i++)
			arr[i] = rand()%256;

		start = chrono::high_resolution_clock::now();

		buffer_A.write(arr.get(), N);

		kernel(cl::NDRange(N), cl::NullRange, buffer_A, buffer_output);

		buffer_output.read(output.get(), N);

		for (size_t i=0; i<N; i++)
			assert(fabs(output[i]-arr[i]-1)<1e-6);

		end = chrono::high_resolution_clock::now();
	}

	// Only use the last result
	return chrono::duration_cast<chrono::microseconds>(end-start).count();
}

uint64_t bench_mapped(size_t N)
{
	unique_ptr<cl_float[]> arr(new cl_float[N]);

	SimpleCLContext context("../speed_test.cl");

	SimpleCLKernel kernel(context.createKernel("increment"));
	SimpleCLBuffer<cl_float> buffer_A = context.createBuffer<cl_float>(N, SimpleCLRead|SimpleCLHostAlloc); // In a real application, we can alloate the buffers once and reuse them over and over again. So the buffer allocation aren't included in the measure the here.
	SimpleCLBuffer<cl_float> buffer_output = context.createBuffer<cl_float>(N, SimpleCLWrite|SimpleCLHostAlloc);

	TimePoint start, end;

	for (int cycle=0; cycle<CYCLE_COUNT; cycle++)
	{
		for (size_t i=0; i<N; i++)
			arr[i] = rand()%256;

		start = chrono::high_resolution_clock::now();

		SimpleCLMappedBuffer<cl_float> A = buffer_A.map();
		memcpy(A.get(), arr.get(), N*sizeof(cl_float));
		A.unmap();

		kernel(cl::NDRange(N), cl::NullRange, buffer_A, buffer_output);

		SimpleCLMappedBuffer<cl_float> output = buffer_output.map(SimpleCLRead);
		for (size_t i=0; i<N; i++)
			assert(fabs(output[i]-arr[i]-1)<1e-4);
		output.unmap();

		end = chrono::high_resolution_clock::now();
	}

	// Only use the last result
	return chrono::duration_cast<chrono::microseconds>(end-start).count();
}

int main()
{
	srand(time(NULL));

	for (size_t N=1; N<1000000; N*=2)
	{
		uint64_t us_mapped = bench_mapped(N);
		uint64_t us_copy = bench_copy(N);
		cout << N << "\t|mapped: " << us_mapped << ", copy:\t" << us_copy << " us, mapped/copy: " << ((double)us_mapped/us_copy) << endl;
		sleep(1);
	}
	return 0;
}

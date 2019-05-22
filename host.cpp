#include <iostream>
#include <assert.h>

#include "SimpleCL.h"

using namespace std;
 
int main()
{
	int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
 	int C[10];

	SimpleCLContext context("host.cl");
  
	cl::Buffer buffer_A = context.createInitBuffer(sizeof(int)*10, A, SimpleCLReadOnly);
	cl::Buffer buffer_B = context.createInitBuffer(sizeof(int)*10, B, SimpleCLReadOnly);
	cl::Buffer buffer_C = context.createBuffer(sizeof(int)*10);

	context.run("simple_add", cl::NDRange(10), buffer_A, buffer_B, buffer_C);

	context.readBuffer(C, buffer_C, sizeof(int)*10);
 
	std::cout<<" result: \n";
	for(int i=0;i<10;i++){
		std::cout<<C[i]<<" ";
	}
 
	return 0;
}

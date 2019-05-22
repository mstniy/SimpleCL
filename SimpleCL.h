#ifndef SIMPLECL_H
#define SIMPLECL_H

#include <CL/cl.hpp>
#include <string>
#include <tuple>

enum SimpleCLMemType
{
	SimpleCLReadOnly,
	SimpleCLWriteOnly,
	SimpleCLReadWrite
};

class SimpleCLContext
{
public:
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
private:
	static void setArgs(cl::Kernel& kernel, int totalCount);
	template<typename T, typename... Args> static void setArgs(cl::Kernel& kernel, int totalCount, const T& arg, const Args&... args);
	static cl_mem_flags smt2cmf(SimpleCLMemType type);
public:
	SimpleCLContext();
	SimpleCLContext(const char* filename);
	cl::Buffer createBuffer(size_t size, SimpleCLMemType type = SimpleCLReadWrite);
	cl::Buffer createInitBuffer(size_t size, void* host_ptr, SimpleCLMemType type = SimpleCLReadWrite);
	void readBuffer(void* host_ptr, const cl::Buffer& buffer, size_t size);
	template<typename... Args> void run(const char* kernelName, const cl::NDRange& range, const Args&... args);
};

#include "SimpleCL.hpp"

#endif

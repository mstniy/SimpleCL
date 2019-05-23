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

class SimpleCLKernel;

class SimpleCLContext
{
public:
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
private:
	static cl_mem_flags smt2cmf(SimpleCLMemType type);
public:
	SimpleCLContext() = default;
	SimpleCLContext(const char* filename);
	cl::Buffer createBuffer(size_t size, SimpleCLMemType type = SimpleCLReadWrite);
	cl::Buffer createInitBuffer(size_t size, void* host_ptr, SimpleCLMemType type = SimpleCLReadWrite);
	void readBuffer(void* host_ptr, const cl::Buffer& buffer, size_t size);
	void writeBuffer(const cl::Buffer& buffer, void* host_ptr, size_t size);
	SimpleCLKernel createKernel(const char* kernelName);
	bool isNull() const;
};

class SimpleCLKernel
{
public:
	cl::Kernel clkernel;
	cl::CommandQueue queue;
private:
	SimpleCLKernel() = default;
	SimpleCLKernel(cl::Kernel _clkernel, cl::CommandQueue _queue);
	void setArgs(int totalCount);
	template<typename T, typename... Args> void setArgs(int totalCount, const T& arg, const Args&... args);
public:
	template<typename... Args> void operator()(const cl::NDRange& range, const Args&... args);

	friend SimpleCLContext;
};

#include "SimpleCL.hpp"

#endif

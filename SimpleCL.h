#ifndef SIMPLECL_H
#define SIMPLECL_H

#include <CL/cl.hpp>
#include <string>
#include <tuple>

typedef int SimpleCLMemType;

const SimpleCLMemType SimpleCLRead = 1;
const SimpleCLMemType SimpleCLWrite = 2;
const SimpleCLMemType SimpleCLReadWrite = 3;
const SimpleCLMemType SimpleCLHostAlloc = 4;

class SimpleCLKernel;

template<typename T>
class SimpleCLBuffer;

class SimpleCLContext
{
public:
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
private:
	static cl_mem_flags smt2cmf(SimpleCLMemType type);
public:
	SimpleCLContext() = default;
	SimpleCLContext(const char* filename);
	template<typename T>
	SimpleCLBuffer<T> createBuffer(size_t length, SimpleCLMemType type = SimpleCLReadWrite);
	template<typename T>
	SimpleCLBuffer<T> createInitBuffer(size_t length, void* host_ptr, SimpleCLMemType type = SimpleCLReadWrite);
	SimpleCLKernel createKernel(const char* kernelName);
	bool isNull() const;
	void finish();
};

template<typename T>
class SimpleCLBuffer
{
public:
	cl::CommandQueue queue;
	cl::Buffer buffer;
	size_t allLength;
	size_t mapCount=0;
public:
	SimpleCLBuffer() = default;
private:
	SimpleCLBuffer(cl::CommandQueue _queue, cl::Buffer _buffer, size_t _allLength);
public:
	void read(void* host_ptr, size_t length);
	void write(const void* host_ptr, size_t length);
	T* map(size_t length, SimpleCLMemType type = SimpleCLReadWrite);
	T* map(SimpleCLMemType type = SimpleCLReadWrite); // This overload maps the entire buffer
	void unmap(T*& ptr);
	size_t length() const;

	friend SimpleCLContext;
};

template<typename T>
class SimpleCLLocalMemory
{
public:
	size_t length;
public:
	SimpleCLLocalMemory(size_t _length);
};

class SimpleCLKernel
{
public:
	cl::Device device;
	cl::Kernel clkernel;
	cl::CommandQueue queue;
private:
	SimpleCLKernel(cl::Device _device, cl::Kernel _clkernel, cl::CommandQueue _queue);
	void setArgs(int totalCount);
	template<typename T, typename... Args> void setArgs(int totalCount, const SimpleCLLocalMemory<T>& arg, const Args&... args);
	template<typename T, typename... Args> void setArgs(int totalCount, const SimpleCLBuffer<T>& arg, const Args&... args);
	template<typename T, typename... Args> void setArgs(int totalCount, const T& arg, const Args&... args);
public:
	SimpleCLKernel() = default;
	template<typename... Args> void runAsync(const cl::NDRange& globalRange, const cl::NDRange& localRange, const Args&... args);
	template<typename... Args> void operator()(const cl::NDRange& globalRange, const cl::NDRange& localRange, const Args&... args);
	size_t getMaxWorkGroupSize() const;

	friend SimpleCLContext;
};

#include "SimpleCL.hpp"

#endif

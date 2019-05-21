#ifndef SIMPLECL_H
#define SIMPLECL_H

#include <CL/cl.hpp>
#include <string>
#include <tuple>

class SimpleCLContext
{
public:
	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program::Sources sources;
	cl::Program program;
	cl::Kernel kernel;
private:
	void setArgsPrivate(int totalCount);
	template<typename T, typename... Args> void setArgsPrivate(int totalCount, const T& arg, const Args&... args);
public:
	SimpleCLContext();
	SimpleCLContext(const std::string& code, const std::string& kernelName);
	SimpleCLContext(const SimpleCLContext&) = delete;
	cl::Buffer createBuffer(size_t size, bool readOnly=false);
	cl::Buffer createInitBuffer(size_t size, void* host_ptr, bool readOnly=false);
	template<typename... Args> void setArgs(const Args&... args);
};

#include "SimpleCL.hpp"

#endif

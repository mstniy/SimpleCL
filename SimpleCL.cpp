#include "SimpleCL.h"
#include <iostream>
#include <assert.h>

cl_mem_flags SimpleCLContext::smt2cmf(SimpleCLMemType type)
{
	if (type == SimpleCLReadOnly)
		return CL_MEM_READ_ONLY;
	else if (type == SimpleCLWriteOnly)
		return CL_MEM_WRITE_ONLY;
	else if (type == SimpleCLReadWrite)
		return CL_MEM_READ_WRITE;
}

SimpleCLContext::SimpleCLContext(const std::string& code)
{
	std::vector<cl::Platform> all_platforms;
	cl_int err;
	err = cl::Platform::get(&all_platforms);
	if (err != CL_SUCCESS)
		throw "cl::Platform::get failed with error code " + std::to_string(err);
	if(all_platforms.size()==0)
		throw "No platforms found. Check OpenCL installation!";
	platform=all_platforms[0];
	std::cout << "Using platform: "<<platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;
	 
	//get default device of the default platform
	std::vector<cl::Device> all_devices;
	err = platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (err != CL_SUCCESS)
		throw "cl::Platform::getDevices failed with error code " + std::to_string(err);
	if(all_devices.size()==0)
		throw "No devices found. Check OpenCL installation!";
	device=all_devices[0];
	std::cout<< "Using device: "<<device.getInfo<CL_DEVICE_NAME>()<<std::endl;
	 
	context = cl::Context({device}, NULL, NULL, NULL, &err);
	if (err != CL_SUCCESS)
		throw "cl::Context constructor failed with error code " + std::to_string(err);
	queue = cl::CommandQueue(context,device, 0, &err);
	if (err != CL_SUCCESS)
		throw "cl::CommandQueue constructor failed with error code " + std::to_string(err);

	sources.push_back({code.c_str(),code.length()});

	program = cl::Program(context,sources, &err);
	if (err != CL_SUCCESS)
		throw "cl::Program constructor failed with error code " + std::to_string(err);
	if(program.build({device}) != CL_SUCCESS)
		throw "Error building: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
}

cl::Buffer SimpleCLContext::createBuffer(size_t size, SimpleCLMemType type)
{
	cl_int err;
	cl_mem_flags flags=smt2cmf(type);
	cl::Buffer buffer(context, flags, size, NULL, &err);
	if (err != CL_SUCCESS)
		throw "cl::Buffer constructor failed with error code " + std::to_string(err);

	return buffer;
}

cl::Buffer SimpleCLContext::createInitBuffer(size_t size, void* host_ptr, SimpleCLMemType type)
{
	cl_int err;
	cl_mem_flags flags=smt2cmf(type) | CL_MEM_COPY_HOST_PTR;
	cl::Buffer buffer(context, flags, size, host_ptr, &err);
	if (err != CL_SUCCESS)
		throw "cl::Buffer constructor failed with error code " + std::to_string(err);

	return buffer;
}

void SimpleCLContext::setArgs(cl::Kernel& kernel, int totalCount)
{
}

void SimpleCLContext::readBuffer(void* host_ptr, const cl::Buffer& buffer, size_t size)
{
	cl_int err;
	err = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, host_ptr);
	if (err != CL_SUCCESS)
		throw "cl::CommandQueue::enqueueReadBuffer failed with error code " + std::to_string(err);
}

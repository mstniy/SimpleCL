#include "SimpleCL.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <sstream>

cl_mem_flags SimpleCLContext::smt2cmf(SimpleCLMemType type)
{
	if (type == SimpleCLReadOnly)
		return CL_MEM_READ_ONLY;
	else if (type == SimpleCLWriteOnly)
		return CL_MEM_WRITE_ONLY;
	else if (type == SimpleCLReadWrite)
		return CL_MEM_READ_WRITE;
}

SimpleCLContext::SimpleCLContext(const char* filename)
{
	std::vector<cl::Platform> all_platforms;
	cl_int err;
	err = cl::Platform::get(&all_platforms);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Platform::get failed with error code " + std::to_string(err));
	if(all_platforms.size()==0)
		throw std::runtime_error("No platforms found. Check OpenCL installation!");
	cl::Platform platform(all_platforms[0]);
	std::cout << "Using platform: "<<platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;
	 
	//get default device of the default platform
	std::vector<cl::Device> all_devices;
	err = platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Platform::getDevices failed with error code " + std::to_string(err));
	if(all_devices.size()==0)
		throw std::runtime_error("No devices found. Check OpenCL installation!");
	device = all_devices[0];
	std::cout<< "Using device: "<<device.getInfo<CL_DEVICE_NAME>()<<std::endl;
	 
	context = cl::Context({device}, NULL, NULL, NULL, &err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Context constructor failed with error code " + std::to_string(err));
	queue = cl::CommandQueue(context,device, 0, &err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue constructor failed with error code " + std::to_string(err));

	std::ifstream file(filename); // From https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
	if (file.is_open() == false)
		throw std::runtime_error(std::string("Failed to open file \"") + filename + "\"");
	std::stringstream buffer;
	buffer << file.rdbuf();
	cl::Program::Sources sources;
	std::string source(buffer.str()); // OpenCL reads garbage during build without this line. No idea why.
	sources.push_back({source.c_str(), source.length()});

	program = cl::Program(context,sources, &err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Program constructor failed with error code " + std::to_string(err));
	if(program.build({device}) != CL_SUCCESS)
		throw std::runtime_error(std::runtime_error("Error building: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)));
}

cl::Buffer SimpleCLContext::createBuffer(size_t size, SimpleCLMemType type)
{
	cl_int err;
	cl_mem_flags flags=smt2cmf(type);
	cl::Buffer buffer(context, flags, size, NULL, &err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Buffer constructor failed with error code " + std::to_string(err));

	return buffer;
}

cl::Buffer SimpleCLContext::createInitBuffer(size_t size, void* host_ptr, SimpleCLMemType type)
{
	cl_int err;
	cl_mem_flags flags=smt2cmf(type) | CL_MEM_COPY_HOST_PTR;
	cl::Buffer buffer(context, flags, size, host_ptr, &err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Buffer constructor failed with error code " + std::to_string(err));

	return buffer;
}

void SimpleCLContext::readBuffer(void* host_ptr, const cl::Buffer& buffer, size_t size)
{
	cl_int err;
	err = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, host_ptr);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueReadBuffer failed with error code " + std::to_string(err));
}

void SimpleCLContext::writeBuffer(const cl::Buffer& buffer, void* host_ptr, size_t size)
{
	cl_int err;
	err = queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size, host_ptr);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueWriteBuffer failed with error code " + std::to_string(err));
}

SimpleCLKernel SimpleCLContext::createKernel(const char* kernelName)
{
	cl_int err;
	cl::Kernel kernel(program, kernelName, &err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Kernel constructor failed with error code " + std::to_string(err));
	return SimpleCLKernel(device, kernel, queue);
}

bool SimpleCLContext::isNull() const
{
	return context() == NULL;
}

void SimpleCLContext::finish()
{
	cl_int err;
	err = queue.finish();
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::finish failed with error code " + std::to_string(err));
}

SimpleCLKernel::SimpleCLKernel(cl::Device _device, cl::Kernel _clkernel, cl::CommandQueue _queue):
	device(_device),
	clkernel(_clkernel),
	queue(_queue)
{
}

void SimpleCLKernel::setArgs(int totalCount)
{
}

size_t SimpleCLKernel::getMaxWorkGroupSize() const
{
	size_t size;
	cl_int err;
	err = clkernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &size);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Kernel::getWorkGroupInfo failed with error code " + std::to_string(err));
	return size;
}
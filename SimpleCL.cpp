#include "SimpleCL.h"
#include <iostream>
#include <assert.h>

SimpleCLContext::SimpleCLContext(const std::string& code, const std::string& kernelName)
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

	kernel = cl::Kernel(program,kernelName.c_str(), &err);
	if (err != CL_SUCCESS)
		throw "cl::Kernel constructor failed with error code " + std::to_string(err);
}

cl::Buffer SimpleCLContext::createBuffer(size_t size, bool readOnly)
{
	cl_int err;
	cl::Buffer buffer(context,readOnly?CL_MEM_READ_ONLY:CL_MEM_READ_WRITE,size, NULL, &err);
	if (err != CL_SUCCESS)
		throw "cl::Buffer constructor failed with error code " + std::to_string(err);

	return buffer;
}

cl::Buffer SimpleCLContext::createInitBuffer(size_t size, void* host_ptr, bool readOnly)
{
	cl_int err;
	cl::Buffer buffer(context,CL_MEM_COPY_HOST_PTR|(readOnly?CL_MEM_READ_ONLY:CL_MEM_READ_WRITE),size, host_ptr, &err);
	if (err != CL_SUCCESS)
		throw "cl::Buffer constructor failed with error code " + std::to_string(err);

	return buffer;
}

void SimpleCLContext::setArgsPrivate(int totalCount)
{
}

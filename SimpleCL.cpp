#include "SimpleCL.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <sstream>

cl_mem_flags SimpleCLContext::smt2cmf(SimpleCLMemType type)
{
	cl_mem_flags flags=0;
	if ((type & SimpleCLRead) && (type&SimpleCLWrite)==0)
		flags = CL_MEM_READ_ONLY;
	else if ((type & SimpleCLRead)==0 && (type&SimpleCLWrite))
		flags = CL_MEM_WRITE_ONLY;
	else if ((type & SimpleCLRead) && (type&SimpleCLWrite))
		flags = CL_MEM_READ_WRITE;
	if (type&SimpleCLHostAlloc)
		flags |= CL_MEM_ALLOC_HOST_PTR;
	return flags;
}

SimpleCLContext::SimpleCLContext(const char* filename, const char* options)
{
	const char* deviceName = getenv("SIMPLECL_DEVICE_NAME");
	std::vector<cl::Platform> all_platforms;
	cl_int err;
	err = cl::Platform::get(&all_platforms);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Platform::get failed with error code " + std::to_string(err));
	if(all_platforms.size()==0)
		throw std::runtime_error("No platforms found. Check OpenCL installation!");
	cl::Platform platform;
	if (deviceName == nullptr)
	{
		platform = all_platforms[0];
		 
		//get default device of the default platform
		std::vector<cl::Device> all_devices;
		err = platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
		if (err != CL_SUCCESS)
			throw std::runtime_error("cl::Platform::getDevices failed with error code " + std::to_string(err));
		if(all_devices.size()==0)
			throw std::runtime_error("No devices found. Check OpenCL installation!");
		device = all_devices[0];
	}
	else
	{
		bool deviceFound = false;
		for (cl::Platform& curPlatform : all_platforms)
		{
			std::vector<cl::Device> devices;
			err = curPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			if (err != CL_SUCCESS)
				throw std::runtime_error("cl::Platform::getDevices failed with error code " + std::to_string(err));
			for (cl::Device& curDevice : devices)
			{
				if (curDevice.getInfo<CL_DEVICE_NAME>().find(deviceName) != std::string::npos)
				{
					deviceFound = true;
					platform = curPlatform;
					device = curDevice;
					break;
				}
			}
			if (deviceFound)
				break;
		}
		if (deviceFound == false)
			throw std::runtime_error(std::string("Failed to find the specified device ") + deviceName);
	}
	std::cout << "Using platform: "<<platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;
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
	if(program.build({device}, options) != CL_SUCCESS)
		throw std::runtime_error(std::runtime_error("Error building: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)));
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

/*void SimpleCLContext::finish()
{
	cl_int err;
	err = queue.finish();
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::finish failed with error code " + std::to_string(err));
}*/

SimpleCLKernel::SimpleCLKernel(cl::Device _device, cl::Kernel _clkernel, cl::CommandQueue _queue):
	device(_device),
	clkernel(_clkernel),
	queue(_queue)
{
}

void SimpleCLKernel::setArgs(int /*totalCount*/)
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
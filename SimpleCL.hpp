template<typename... Args>
void SimpleCLContext::run(const char* kernelName, const cl::NDRange& range, const Args&... args)
{
	cl_int err;
	cl::Kernel kernel(program,kernelName, &err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Kernel constructor failed with error code " + std::to_string(err));
	setArgs(kernel, sizeof...(args), args...);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueNDRangeKernel failed with error code " + std::to_string(err));
	err = queue.finish();
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::finish failed with error code " + std::to_string(err));
}

template<typename T, typename... Args> 
void SimpleCLContext::setArgs(cl::Kernel& kernel, int totalCount, const T& arg, const Args&... args)
{
	kernel.setArg(totalCount-sizeof...(args)-1, arg);
	setArgs(kernel, totalCount, args...);
}

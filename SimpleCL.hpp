template<typename... Args>
void SimpleCLContext::run(const cl::NDRange& range, const Args&... args)
{
	cl_int err;
	setArgs(sizeof...(args), args...);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
	if (err != CL_SUCCESS)
		throw "cl::CommandQueue::enqueueNDRangeKernel failed with error code " + std::to_string(err);
	err = queue.finish();
	if (err != CL_SUCCESS)
		throw "cl::CommandQueue::finish failed with error code " + std::to_string(err);
}

template<typename T, typename... Args> 
void SimpleCLContext::setArgs(int totalCount, const T& arg, const Args&... args)
{
	kernel.setArg(totalCount-sizeof...(args)-1, arg);
	setArgs(totalCount, args...);
}

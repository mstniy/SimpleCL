template<typename... Args>
void SimpleCLKernel::operator()(const cl::NDRange& range, const Args&... args)
{
	cl_int err;
	setArgs(sizeof...(args), args...);
	err = queue.enqueueNDRangeKernel(clkernel, cl::NullRange, range, cl::NullRange);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueNDRangeKernel failed with error code " + std::to_string(err));
	err = queue.finish();
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::finish failed with error code " + std::to_string(err));
}

template<typename T, typename... Args> 
void SimpleCLKernel::setArgs(int totalCount, const T& arg, const Args&... args)
{
	clkernel.setArg(totalCount-sizeof...(args)-1, arg);
	setArgs(totalCount, args...);
}

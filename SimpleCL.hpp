template<typename T>
SimpleCLLocalMemory<T>::SimpleCLLocalMemory(size_t _size):
	size(_size)
{
}

template<typename... Args>
void SimpleCLKernel::runAsync(const cl::NDRange& globalRange, const cl::NDRange& localRange, const Args&... args)
{
	cl_int err;
	setArgs(sizeof...(args), args...);
	err = queue.enqueueNDRangeKernel(clkernel, cl::NullRange, globalRange, localRange);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueNDRangeKernel failed with error code " + std::to_string(err));
}

template<typename... Args>
void SimpleCLKernel::operator()(const cl::NDRange& globalRange, const cl::NDRange& localRange, const Args&... args)
{
	runAsync(globalRange, localRange, args...);
	cl_int err;
	err = queue.finish();
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::finish failed with error code " + std::to_string(err));
}

template<typename T, typename... Args> 
void SimpleCLKernel::setArgs(int totalCount, const SimpleCLLocalMemory<T>& arg, const Args&... args)
{
	clkernel.setArg(totalCount-sizeof...(args)-1, arg.size*sizeof(T), NULL);
	setArgs(totalCount, args...);
}

template<typename T, typename... Args> 
void SimpleCLKernel::setArgs(int totalCount, const T& arg, const Args&... args)
{
	clkernel.setArg(totalCount-sizeof...(args)-1, arg);
	setArgs(totalCount, args...);
}

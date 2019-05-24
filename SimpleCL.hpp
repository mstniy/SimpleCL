template<typename T>
SimpleCLLocalMemory<T>::SimpleCLLocalMemory(size_t _size):
	size(_size)
{
}

template<typename T>
SimpleCLBuffer<T> SimpleCLContext::createBuffer(size_t length, SimpleCLMemType type)
{
	cl_int err;
	cl_mem_flags flags=smt2cmf(type);
	cl::Buffer buffer(context, flags, length*sizeof(T), NULL, &err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Buffer constructor failed with error code " + std::to_string(err));

	return SimpleCLBuffer<T>(queue, buffer, length);
}

template<typename T>
SimpleCLBuffer<T> SimpleCLContext::createInitBuffer(size_t length, void* host_ptr, SimpleCLMemType type)
{
	cl_int err;
	cl_mem_flags flags=smt2cmf(type) | CL_MEM_COPY_HOST_PTR;
	cl::Buffer buffer(context, flags, length*sizeof(T), host_ptr, &err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Buffer constructor failed with error code " + std::to_string(err));

	return SimpleCLBuffer<T>(queue, buffer, length);
}

template<typename T>
SimpleCLBuffer<T>::SimpleCLBuffer(cl::CommandQueue _queue, cl::Buffer _buffer, size_t _allLength):
	queue(_queue),
	buffer(_buffer),
	allLength(_allLength)
{
}

template<typename T>
void SimpleCLBuffer<T>::read(void* host_ptr, size_t length)
{
	if (mapped)
		throw std::runtime_error("Cannot read from mapped buffer");
	cl_int err;
	err = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, length*sizeof(T), host_ptr);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueReadBuffer failed with error code " + std::to_string(err));
}

template<typename T>
void SimpleCLBuffer<T>::write(const void* host_ptr, size_t length)
{
	if (mapped)
		throw std::runtime_error("Cannot write to mapped buffer");
	cl_int err;
	err = queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, length*sizeof(T), host_ptr);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueWriteBuffer failed with error code " + std::to_string(err));
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
void SimpleCLKernel::setArgs(int totalCount, const SimpleCLBuffer<T>& arg, const Args&... args)
{
	clkernel.setArg(totalCount-sizeof...(args)-1, arg.buffer);
	setArgs(totalCount, args...);
}

template<typename T, typename... Args> 
void SimpleCLKernel::setArgs(int totalCount, const T& arg, const Args&... args)
{
	clkernel.setArg(totalCount-sizeof...(args)-1, arg);
	setArgs(totalCount, args...);
}

template<typename T>
SimpleCLLocalMemory<T>::SimpleCLLocalMemory(size_t _length):
	length(_length)
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
	if (mapCount() != 0)
		throw std::runtime_error("Cannot enqueueReadBuffer on mapped buffer");
	cl_int err;
	err = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, length*sizeof(T), host_ptr);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueReadBuffer failed with error code " + std::to_string(err));
}

template<typename T>
void SimpleCLBuffer<T>::write(const void* host_ptr, size_t length)
{
	if (mapCount() != 0)
		throw std::runtime_error("Cannot enqueueWriteBuffer on mapped buffer");
	cl_int err;
	err = queue.enqueueWriteBuffer(buffer, CL_FALSE, 0, length*sizeof(T), host_ptr); // We can use non-blocking write here, since we're not using out-of-order command queues.
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueWriteBuffer failed with error code " + std::to_string(err));
}

template<typename T>
size_t SimpleCLBuffer<T>::mapCount() const
{
	cl_int err;
	size_t res = buffer.getInfo<CL_MEM_MAP_COUNT>(&err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::Buffer::getInfo failed with error code " + std::to_string(err));
	return res;
}

template<typename T>
SimpleCLMappedBuffer<T> SimpleCLBuffer<T>::map(size_t length, SimpleCLMemType type)
{
	cl_map_flags flags=0;
	if (type & SimpleCLRead)
		flags |= CL_MAP_READ;
	if (type & SimpleCLWrite)
		flags |= CL_MAP_WRITE;
	cl_int err;
	void* ptr = queue.enqueueMapBuffer(buffer, CL_TRUE, flags, 0, length*sizeof(T), NULL, NULL, &err);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueMapBuffer failed with error code " + std::to_string(err));
	return SimpleCLMappedBuffer<T>(queue, buffer, (T*)ptr);
}

template<typename T>
SimpleCLMappedBuffer<T> SimpleCLBuffer<T>::map(SimpleCLMemType type)
{
	return map(allLength, type);
}

template<typename T>
T& SimpleCLMappedBuffer<T>::operator[](size_t i)
{
	return map[i];
}

template<typename T>
const T& SimpleCLMappedBuffer<T>::operator[](size_t i) const
{
	return map[i];
}

template<typename T>
T* SimpleCLMappedBuffer<T>::get()
{
	return map;
}

template<typename T>
const T* SimpleCLMappedBuffer<T>::get() const
{
	return map;
}

template<typename T>
size_t SimpleCLBuffer<T>::length() const
{
	return allLength;
}

template<typename T>
SimpleCLMappedBuffer<T>::SimpleCLMappedBuffer(cl::CommandQueue _queue, cl::Buffer _buffer, T* _map):
	queue(_queue),
	buffer(_buffer),
	map(_map)
{
}

template<typename T>
SimpleCLMappedBuffer<T>::SimpleCLMappedBuffer(SimpleCLMappedBuffer&& o):
	queue(o.queue),
	buffer(o.buffer),
	map(o.map)
{
	o.map = nullptr;
}

template<typename T>
SimpleCLMappedBuffer<T>& SimpleCLMappedBuffer<T>::operator=(SimpleCLMappedBuffer&& o)
{
	queue = o.queue;
	buffer = o.buffer;
	map = o.map;
	o.map = nullptr;
	return (*this);
}

template<typename T>
SimpleCLMappedBuffer<T>::~SimpleCLMappedBuffer()
{
	unmap();
}

template<typename T>
void SimpleCLMappedBuffer<T>::unmap()
{
	if (map == nullptr) // Already empty
		return ;
	cl_int err;
	err = queue.enqueueUnmapMemObject(buffer, (void*)map);
	if (err != CL_SUCCESS)
		throw std::runtime_error("cl::CommandQueue::enqueueUnmapMemObject failed with error code " + std::to_string(err));
	map = nullptr;
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
	clkernel.setArg(totalCount-sizeof...(args)-1, arg.length*sizeof(T), NULL);
	setArgs(totalCount, args...);
}

template<typename T, typename... Args>
void SimpleCLKernel::setArgs(int totalCount, const SimpleCLBuffer<T>& arg, const Args&... args)
{
	if (arg.mapCount() != 0)
		throw std::runtime_error("Buffers must be unmapped before they are passed to a kernel");
	clkernel.setArg(totalCount-sizeof...(args)-1, arg.buffer);
	setArgs(totalCount, args...);
}

template<typename T, typename... Args> 
void SimpleCLKernel::setArgs(int totalCount, const T& arg, const Args&... args)
{
	clkernel.setArg(totalCount-sizeof...(args)-1, arg);
	setArgs(totalCount, args...);
}

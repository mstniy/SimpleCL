template<typename... Args>
void SimpleCLContext::setArgs(const Args&... args)
{
	setArgsPrivate(sizeof...(args), args...);
}

template<typename T, typename... Args> 
void SimpleCLContext::setArgsPrivate(int totalCount, const T& arg, const Args&... args)
{
	kernel.setArg(totalCount-sizeof...(args)-1, arg);
	setArgsPrivate(totalCount, args...);
}

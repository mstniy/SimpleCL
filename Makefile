host.exe: host.cpp SimpleCL.cpp SimpleCL.h SimpleCL.hpp
	g++ host.cpp SimpleCL.cpp -o $@ --std=c++11 -l OpenCL -I /usr/local/cuda-10.1/include -L /usr/local/cuda-10.1/targets/x86_64-linux/lib/

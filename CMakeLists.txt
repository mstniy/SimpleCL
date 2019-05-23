cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(simplecl_demo)
set(CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_STANDARD 11)

include_directories(SYSTEM "/usr/local/cuda-10.1/include")
link_directories("/usr/local/cuda-10.1/targets/x86_64-linux/lib/")
add_compile_options(-Wall -Wextra -pedantic)

add_executable(add_demo add_demo.cpp SimpleCL.cpp)
target_link_libraries(add_demo OpenCL)

add_executable(sum_demo sum_demo.cpp SimpleCL.cpp)
target_link_libraries(sum_demo OpenCL)
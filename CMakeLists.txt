cmake_minimum_required(VERSION 3.10)
project(nccl_allreduce_demo LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 查找NCCL
find_library(NCCL_LIB nccl PATHS $ENV{NCCL_HOME}/lib $ENV{NCCL_DIR}/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)

if(NOT NCCL_LIB)
  message(FATAL_ERROR "NCCL library not found")
endif()

# 添加头文件目录
include_directories($ENV{NCCL_HOME}/include $ENV{NCCL_DIR}/include /usr/include)

# 收集源文件
file(GLOB SOURCES "src/*.cpp")

# 生成可执行文件
add_executable(nccl_allreduce_demo ${SOURCES})
target_link_libraries(nccl_allreduce_demo ${NCCL_LIB} ${CUDA_LIBRARIES})

# 启用显示调试信息
target_compile_definitions(nccl_allreduce_demo PRIVATE -DDEBUG)
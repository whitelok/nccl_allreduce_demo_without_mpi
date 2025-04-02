#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// CUDA错误检查
#define CUDA_CHECK(cmd)                                                                                    \
  do {                                                                                                     \
    cudaError_t err = cmd;                                                                                 \
    if (err != cudaSuccess) {                                                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")" \
                << std::endl;                                                                              \
      throw std::runtime_error("CUDA error");                                                              \
    }                                                                                                      \
  } while (0)

// NCCL错误检查
#define NCCL_CHECK(cmd)                                                                                    \
  do {                                                                                                     \
    ncclResult_t res = cmd;                                                                                \
    if (res != ncclSuccess) {                                                                              \
      std::cerr << "NCCL error: " << ncclGetErrorString(res) << " (" << __FILE__ << ":" << __LINE__ << ")" \
                << std::endl;                                                                              \
      throw std::runtime_error("NCCL error");                                                              \
    }                                                                                                      \
  } while (0)

// 设备选择和初始化辅助函数
class NCCLHelper {
 public:
  static void getLocalRank(int& localRank, int& localSize, int rank, int worldSize) {
    // 从环境变量中获取本地信息（适用于大多数作业调度系统）
    char* localRankStr = getenv("SLURM_LOCALID");
    char* localSizeStr = getenv("SLURM_NTASKS_PER_NODE");

    if (localRankStr && localSizeStr) {
      localRank = atoi(localRankStr);
      localSize = atoi(localSizeStr);
    } else {
      // 假设所有进程在同一节点上，或自行计算
      localRank = rank;
      localSize = worldSize;

      // 这里可以添加其他作业调度器的环境变量支持
    }
  }

  static void setDevice(int rank, int localRank) {
    // 根据本地排名选择设备
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount <= 0) {
      throw std::runtime_error("No CUDA devices found");
    }

    int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    std::cout << "Rank " << rank << " (local rank " << localRank << ") using CUDA device " << device << std::endl;
  }

  // 打印GPU信息
  static void printDeviceInfo(int rank) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Rank " << rank << " using " << prop.name << " with " << prop.multiProcessorCount << " SMs and "
              << (prop.totalGlobalMem / (1024 * 1024)) << " MB memory" << std::endl;
  }
};
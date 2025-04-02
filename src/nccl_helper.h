#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <utils.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(cmd)                                                                                    \
  do {                                                                                                     \
    cudaError_t err = cmd;                                                                                 \
    if (err != cudaSuccess) {                                                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")" \
                << std::endl;                                                                              \
      throw std::runtime_error("CUDA error");                                                              \
    }                                                                                                      \
  } while (0)

#define NCCL_CHECK(cmd)                                                                                    \
  do {                                                                                                     \
    ncclResult_t res = cmd;                                                                                \
    if (res != ncclSuccess) {                                                                              \
      std::cerr << "NCCL error: " << ncclGetErrorString(res) << " (" << __FILE__ << ":" << __LINE__ << ")" \
                << std::endl;                                                                              \
      throw std::runtime_error("NCCL error");                                                              \
    }                                                                                                      \
  } while (0)

#define NCCLCHECK(cmd) NCCL_CHECK(cmd)

// class Logger {
//  private:
//   static std::mutex mutex;
//   int nodeRank;

//  public:
//   Logger(int nodeRank = 0) : nodeRank(nodeRank) {}

//   template <typename T>
//   static void log(const T& msg) {
//     std::lock_guard<std::mutex> lock(mutex);
//     std::cout << msg << std::endl;
//   }

//   template <typename T>
//   void nodeLog(const T& msg) {
//     std::lock_guard<std::mutex> lock(mutex);
//     std::cout << "[Node " << nodeRank << "] " << msg << std::endl;
//   }

//   template <typename T, typename... Args>
//   void nodeLog(const T& msg, Args... args) {
//     std::ostringstream oss;
//     oss << msg;
//     nodeLog(oss.str(), args...);
//   }
// };

class NCCLHelper {
 public:
  static int getDeviceCount() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
  }

  static void printDeviceInfo(int deviceId, int globalRank) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    std::ostringstream oss;
    oss << "GPU " << deviceId << " (global rank " << globalRank << "): " << prop.name << " with "
        << prop.multiProcessorCount << " SMs, " << (prop.totalGlobalMem / (1024 * 1024)) << " MB memory";
    Logger::log(oss.str());
  }
};
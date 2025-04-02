#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// CUDA错误检查宏
#define CUDA_CHECK(cmd)                                                                                    \
  do {                                                                                                     \
    cudaError_t err = cmd;                                                                                 \
    if (err != cudaSuccess) {                                                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")" \
                << std::endl;                                                                              \
      throw std::runtime_error("CUDA error");                                                              \
    }                                                                                                      \
  } while (0)

// NCCL错误检查宏
#define NCCL_CHECK(cmd)                                                                                    \
  do {                                                                                                     \
    ncclResult_t res = cmd;                                                                                \
    if (res != ncclSuccess) {                                                                              \
      std::cerr << "NCCL error: " << ncclGetErrorString(res) << " (" << __FILE__ << ":" << __LINE__ << ")" \
                << std::endl;                                                                              \
      throw std::runtime_error("NCCL error");                                                              \
    }                                                                                                      \
  } while (0)

// 线程安全的日志记录器
class Logger {
 private:
  static std::mutex mutex;

 public:
  // 线程安全日志输出
  template <typename T>
  static void log(const T& msg) {
    std::lock_guard<std::mutex> lock(mutex);
    std::cout << msg << std::endl;
  }

  template <typename T, typename... Args>
  static void log(const T& msg, const Args&... args) {
    std::ostringstream oss;
    oss << msg;
    logHelper(oss, args...);
  }

 private:
  template <typename T, typename... Args>
  static void logHelper(std::ostringstream& oss, const T& msg, const Args&... args) {
    oss << " " << msg;
    logHelper(oss, args...);
  }

  static void logHelper(std::ostringstream& oss) { log(oss.str()); }
};

std::mutex Logger::mutex;

// 同步屏障 - 用于线程间同步
class Barrier {
 private:
  std::mutex mutex;
  std::condition_variable cv;
  size_t count;
  size_t threshold;
  size_t generation;

 public:
  explicit Barrier(size_t count) : count(count), threshold(count), generation(0) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    size_t gen = generation;

    if (--count == 0) {
      generation++;
      count = threshold;
      cv.notify_all();
      return;
    }

    cv.wait(lock, [this, gen] { return gen != generation; });
  }
};

// 命令行参数解析
struct AppArgs {
  int nodeRank;          // 当前节点rank
  int nodeCount;         // 总节点数
  std::string masterIP;  // 主节点IP地址
  int port;              // 通信端口
  size_t dataSize;       // 数据大小(以float元素数量计)
  int iterations;        // 迭代次数(性能测试)

  // 默认构造函数
  AppArgs() : nodeRank(0), nodeCount(1), masterIP("localhost"), port(9999), dataSize(1000000), iterations(10) {}

  // 从命令行解析参数
  static AppArgs parseArgs(int argc, char* argv[]) {
    AppArgs args;

    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "--rank" && i + 1 < argc)
        args.nodeRank = std::atoi(argv[++i]);
      else if (arg == "--nproc" && i + 1 < argc)
        args.nodeCount = std::atoi(argv[++i]);
      else if (arg == "--master" && i + 1 < argc)
        args.masterIP = argv[++i];
      else if (arg == "--port" && i + 1 < argc)
        args.port = std::atoi(argv[++i]);
      else if (arg == "--size" && i + 1 < argc)
        args.dataSize = std::atoll(argv[++i]);
      else if (arg == "--iter" && i + 1 < argc)
        args.iterations = std::atoi(argv[++i]);
    }

    return args;
  }

  // 打印配置信息
  void printConfig() const {
    Logger::log("Configuration:");
    Logger::log("  Node Rank:", nodeRank);
    Logger::log("  Node Count:", nodeCount);
    Logger::log("  Master IP:", masterIP);
    Logger::log("  Port:", port);
    Logger::log("  Data Size:", dataSize, "elements");
    Logger::log("  Iterations:", iterations);
  }
};
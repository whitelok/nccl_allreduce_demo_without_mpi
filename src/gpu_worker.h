#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <atomic>
#include <thread>
#include <vector>
#include "utils.h"

// GPU工作线程上下文
struct GPUContext {
  int deviceId;         // 本地GPU设备ID
  int localRank;        // 本地线程排名
  int globalRank;       // 全局GPU排名
  int nodeRank;         // 节点排名
  int localSize;        // 本地GPU数量
  int globalSize;       // 全局GPU总数
  size_t dataSize;      // 处理的数据大小
  ncclComm_t comm;      // NCCL通信器
  cudaStream_t stream;  // CUDA流
  float* d_input;       // 输入数据
  float* d_output;      // 输出结果
  bool success;         // 操作成功标志

  // 性能测量
  double avgTimeMs;
  double bandwidth;

  // 初始构造函数
  GPUContext()
      : deviceId(-1),
        localRank(-1),
        globalRank(-1),
        nodeRank(-1),
        localSize(0),
        globalSize(0),
        dataSize(0),
        comm(nullptr),
        stream(nullptr),
        d_input(nullptr),
        d_output(nullptr),
        success(false),
        avgTimeMs(0.0),
        bandwidth(0.0) {}
};

// GPU工作线程的管理器
class GPUWorkerManager {
 public:
  // 构造函数
  GPUWorkerManager(int nodeRank, int nodeCount, size_t dataSize, int iterations);

  // 析构函数
  ~GPUWorkerManager();

  // 初始化并运行GPU线程
  bool runAllreduce(const ncclUniqueId& ncclId);

  // 获取性能结果
  std::vector<GPUContext>& getContexts() { return contexts; }

  // 是否全部成功
  bool isSuccessful() const { return !initError; }

  // 获取本地GPU数量
  int getLocalDeviceCount() const { return deviceCount; }

  // 获取平均执行时间和带宽
  void getPerformanceStats(double& avgTime, double& avgBandwidth) const;

 private:
  // 配置信息
  int nodeRank;
  int nodeCount;
  int deviceCount;
  size_t dataSize;
  int iterations;

  // 线程相关
  std::vector<std::thread> threads;
  std::vector<GPUContext> contexts;
  std::atomic<bool> initError;

  // 线程同步
  std::unique_ptr<Barrier> initBarrier;
  std::unique_ptr<Barrier> syncBarrier;

  // GPU工作线程函数
  static void gpuWorkerThread(GPUContext& ctx, Barrier& initBarrier, Barrier& syncBarrier, std::atomic<bool>& initError,
                              int iterations);
};
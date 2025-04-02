#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "nccl_helper.h"

// 线程同步屏障类
class Barrier {
 private:
  std::mutex mutex;
  std::condition_variable cv;
  std::size_t count;
  std::size_t threshold;
  std::size_t generation;

 public:
  explicit Barrier(std::size_t count) : count(count), threshold(count), generation(0) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    std::size_t gen = generation;

    if (--count == 0) {
      generation++;
      count = threshold;
      cv.notify_all();
      return;
    }

    cv.wait(lock, [this, gen] { return gen != generation; });
  }
};

// GPU线程的工作上下文
struct GPUThreadContext {
  int deviceId;         // 本地GPU设备ID
  int globalRank;       // 全局GPU排名
  int worldSize;        // 总GPU数量
  size_t dataSize;      // 处理的数据大小
  ncclComm_t comm;      // NCCL通信器
  cudaStream_t stream;  // CUDA流
  float* d_data;        // 输入数据
  float* d_result;      // 输出结果
  bool success;         // 操作成功标志
  int nodeRank;         // 节点排名（用于日志）

  // 性能测量
  double avgTimeMs;
  double bandwidth;

  GPUThreadContext()
      : deviceId(-1),
        globalRank(-1),
        worldSize(0),
        dataSize(0),
        comm(nullptr),
        stream(nullptr),
        d_data(nullptr),
        d_result(nullptr),
        success(false),
        nodeRank(0),
        avgTimeMs(0),
        bandwidth(0) {}
};

// GPU线程工作函数
void gpuWorkerThread(GPUThreadContext& ctx, const ncclUniqueId& ncclId, Barrier& initBarrier, Barrier& syncBarrier,
                     std::atomic<bool>& initError, int iterations) {
  Logger logger(ctx.nodeRank);

  try {
    // 设置当前线程的CUDA设备
    CUDA_CHECK(cudaSetDevice(ctx.deviceId));

    // 初始化该设备的CUDA流
    CUDA_CHECK(cudaStreamCreate(&ctx.stream));

    // 打印设备信息
    NCCLHelper::printDeviceInfo(ctx.deviceId, ctx.globalRank);

    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(&ctx.d_data, ctx.dataSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_result, ctx.dataSize * sizeof(float)));

    // 初始化输入数据 - 每个GPU使用不同值 (全局排名+1)
    std::vector<float> h_data(ctx.dataSize, ctx.globalRank + 1.0f);
    CUDA_CHECK(cudaMemcpy(ctx.d_data, h_data.data(), ctx.dataSize * sizeof(float), cudaMemcpyHostToDevice));

    // 在所有线程完成基本初始化前等待
    initBarrier.wait();

    // 初始化NCCL通信器
    logger.nodeLog("GPU ", ctx.deviceId, " initializing NCCL communicator with rank ", ctx.globalRank, " / ",
                   ctx.worldSize);

    NCCL_CHECK(ncclCommInitRank(&ctx.comm, ctx.worldSize, ncclId, ctx.globalRank));
    logger.nodeLog("GPU ", ctx.deviceId, " (global rank ", ctx.globalRank, ") initialized NCCL communicator");

    // 在所有线程完成NCCL初始化前等待
    syncBarrier.wait();

    // 执行验证测试
    NCCL_CHECK(ncclAllReduce(ctx.d_data, ctx.d_result, ctx.dataSize, ncclFloat, ncclSum, ctx.comm, ctx.stream));

    // 等待操作完成
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // 检验结果
    std::vector<float> h_result(ctx.dataSize);
    CUDA_CHECK(cudaMemcpy(h_result.data(), ctx.d_result, ctx.dataSize * sizeof(float), cudaMemcpyDeviceToHost));

    // 计算预期结果：sum(1..worldSize)
    float expectedSum = ctx.worldSize * (ctx.worldSize + 1) / 2.0f;
    bool correct = true;

    // 检查结果（仅检查前几个元素）
    size_t checkLimit = std::min(ctx.dataSize, static_cast<size_t>(10));
    for (size_t i = 0; i < checkLimit; i++) {
      if (std::abs(h_result[i] - expectedSum) > 1e-5) {
                logger.nodeLog("GPU ", ctx.deviceId, " (global rank ", ctx.globalRank, 
                             "): Validation failed at index ", i,
                             ", got ", h_result[i], ", expected ", expectedSum
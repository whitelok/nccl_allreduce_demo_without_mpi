#include "gpu_worker.h"
#include <chrono>
#include <cmath>
#include <iomanip>

GPUWorkerManager::GPUWorkerManager(int nodeRank, int nodeCount, size_t dataSize, int iterations)
    : nodeRank(nodeRank), nodeCount(nodeCount), dataSize(dataSize), iterations(iterations), initError(false) {
  // 获取当前节点上的GPU数量
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

  if (deviceCount <= 0) {
    throw std::runtime_error("No CUDA devices found on this node");
  }

  Logger::log("Node", nodeRank, "found", deviceCount, "GPU devices");

  // 创建同步屏障
  initBarrier = std::make_unique<Barrier>(deviceCount);
  syncBarrier = std::make_unique<Barrier>(deviceCount);

  // 初始化GPU上下文
  contexts.resize(deviceCount);
  for (int i = 0; i < deviceCount; i++) {
    contexts[i].deviceId = i;
    contexts[i].localRank = i;
    contexts[i].nodeRank = nodeRank;
    contexts[i].localSize = deviceCount;
    contexts[i].globalRank = nodeRank * deviceCount + i;
    contexts[i].globalSize = nodeCount * deviceCount;
    contexts[i].dataSize = dataSize;
  }
}

GPUWorkerManager::~GPUWorkerManager() {
  // 等待所有线程完成
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

bool GPUWorkerManager::runAllreduce(const ncclUniqueId& ncclId) {
  // 启动GPU工作线程
  for (int i = 0; i < deviceCount; i++) {
    threads.emplace_back(gpuWorkerThread, std::ref(contexts[i]), std::ref(*initBarrier), std::ref(*syncBarrier),
                         std::ref(initError), iterations, std::ref(ncclId));
  }

  // 等待所有线程完成
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  return !initError;
}

void GPUWorkerManager::getPerformanceStats(double& avgTime, double& avgBandwidth) const {
  avgTime = 0.0;
  avgBandwidth = 0.0;
  int successCount = 0;

  for (const auto& ctx : contexts) {
    if (ctx.success) {
      avgTime += ctx.avgTimeMs;
      avgBandwidth += ctx.bandwidth;
      successCount++;
    }
  }

  if (successCount > 0) {
    avgTime /= successCount;
    avgBandwidth /= successCount;
  }
}

void GPUWorkerManager::gpuWorkerThread(GPUContext& ctx, Barrier& initBarrier, Barrier& syncBarrier,
                                       std::atomic<bool>& initError, int iterations, const ncclUniqueId& ncclId) {
  try {
    // 设置当前线程的CUDA设备
    CUDA_CHECK(cudaSetDevice(ctx.deviceId));

    // 初始化该设备的CUDA流
    CUDA_CHECK(cudaStreamCreate(&ctx.stream));

    // 获取并打印设备信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ctx.deviceId));
    Logger::log("Node", ctx.nodeRank, "GPU", ctx.deviceId, "using", prop.name, "with", prop.multiProcessorCount,
                "SMs and", (prop.totalGlobalMem / (1024 * 1024)), "MB memory,", "Global rank:", ctx.globalRank);

    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(&ctx.d_input, ctx.dataSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_output, ctx.dataSize * sizeof(float)));

    // 初始化输入数据 - 每个GPU使用不同值 (基于全局排名)
    std::vector<float> h_data(ctx.dataSize, ctx.globalRank + 1.0f);
    CUDA_CHECK(cudaMemcpy(ctx.d_input, h_data.data(), ctx.dataSize * sizeof(float), cudaMemcpyHostToDevice));

    // 在所有线程完成初始化前等待
    initBarrier.wait();

    // 初始化NCCL通信器，使用全局排名
    NCCL_CHECK(ncclCommInitRank(&ctx.comm, ctx.globalSize, ncclId, ctx.globalRank));
    Logger::log("Node", ctx.nodeRank, "GPU", ctx.deviceId, "initialized NCCL communicator with global rank",
                ctx.globalRank);

    // 同步所有线程的NCCL初始化
    syncBarrier.wait();

    // 执行验证测试
    NCCL_CHECK(ncclAllReduce(ctx.d_input, ctx.d_output, ctx.dataSize, ncclFloat, ncclSum, ctx.comm, ctx.stream));

    // 等待操作完成
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // 检验结果
    std::vector<float> h_result(ctx.dataSize);
    CUDA_CHECK(cudaMemcpy(h_result.data(), ctx.d_output, ctx.dataSize * sizeof(float), cudaMemcpyDeviceToHost));

    // 计算预期结果：sum(1..globalSize)
    float expectedSum = ctx.globalSize * (ctx.globalSize + 1) / 2.0f;
    bool correct = true;

    // 检查结果（只检查部分数据）
    size_t checkLimit = std::min(ctx.dataSize, static_cast<size_t>(10));
    for (size_t i = 0; i < checkLimit; i++) {
      if (std::abs(h_result[i] - expectedSum) > 1e-5) {
        Logger::log("Node", ctx.nodeRank, "GPU", ctx.deviceId, "validation failed at index", i, "got", h_result[i],
                    "expected", expectedSum);
        correct = false;
        break;
      }
    }

    if (correct) {
      Logger::log("Node", ctx.nodeRank, "GPU", ctx.deviceId, "(global rank", ctx.globalRank,
                  "): AllReduce validation PASSED!");
    }

    // 等待所有线程完成验证
    syncBarrier.wait();

    // 执行AllReduce基准测试
    Logger::log("Node", ctx.nodeRank, "GPU", ctx.deviceId, "running AllReduce benchmark...");

    // 测量多次迭代的平均时间
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
      NCCL_CHECK(ncclAllReduce(ctx.d_input, ctx.d_output, ctx.dataSize, ncclFloat, ncclSum, ctx.comm, ctx.stream));
      CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    ctx.avgTimeMs = (elapsed.count() * 1000) / iterations;

    // 计算带宽 (2*N: 发送+接收, *4: float是4字节)
    ctx.bandwidth =
        (2.0 * ctx.dataSize * sizeof(float) * ctx.globalSize) / (ctx.avgTimeMs / 1000) / (1024 * 1024 * 1024);

    Logger::log("Node", ctx.nodeRank, "GPU", ctx.deviceId, "benchmark complete:", std::fixed, std::setprecision(3),
                ctx.avgTimeMs, "ms,", std::setprecision(2), ctx.bandwidth, "GB/s");

    // 同步所有线程的基准测试
    syncBarrier.wait();

    // 设置成功标志
    ctx.success = true;

  } catch (const std::exception& e) {
    Logger::log("Error in GPU thread (Node", ctx.nodeRank, "GPU", ctx.deviceId, "):", e.what());
    ctx.success = false;
    initError = true;
  }

  // 清理资源
  try {
    if (ctx.d_input) CUDA_CHECK(cudaFree(ctx.d_input));
    if (ctx.d_output) CUDA_CHECK(cudaFree(ctx.d_output));
    if (ctx.stream) CUDA_CHECK(cudaStreamDestroy(ctx.stream));
    if (ctx.comm) NCCL_CHECK(ncclCommDestroy(ctx.comm));
  } catch (...) {
    // 清理过程中的错误不再抛出
  }
}
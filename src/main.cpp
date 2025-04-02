#include <unistd.h>  // gethostname
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#include "nccl_helper.h"
#include "tcp_socket.h"

// 参数解析函数
void parseArgs(int argc, char* argv[], int& rank, int& worldSize, std::string& masterIP, int& port, size_t& dataSize) {
  rank = 0;
  worldSize = 1;
  masterIP = "localhost";
  port = 9999;
  dataSize = 1000000;  // 默认1M元素

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--rank" && i + 1 < argc)
      rank = std::atoi(argv[++i]);
    else if (arg == "--world-size" && i + 1 < argc)
      worldSize = std::atoi(argv[++i]);
    else if (arg == "--master" && i + 1 < argc)
      masterIP = argv[++i];
    else if (arg == "--port" && i + 1 < argc)
      port = std::atoi(argv[++i]);
    else if (arg == "--size" && i + 1 < argc)
      dataSize = std::atoll(argv[++i]);
  }

  // 打印配置信息
  if (rank == 0) {
    std::cout << "Configuration:" << std::endl
              << "  World Size: " << worldSize << std::endl
              << "  Master IP: " << masterIP << std::endl
              << "  Port: " << port << std::endl
              << "  Data Size: " << dataSize << " elements" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  try {
    int rank, worldSize;
    std::string masterIP;
    int port;
    size_t dataSize;

    // 解析命令行参数
    parseArgs(argc, argv, rank, worldSize, masterIP, port, dataSize);

    // 获取本地排名并设置设备
    int localRank, localSize;
    NCCLHelper::getLocalRank(localRank, localSize, rank, worldSize);
    NCCLHelper::setDevice(rank, localRank);
    NCCLHelper::printDeviceInfo(rank);

    // 在master上创建唯一NCCL ID
    ncclUniqueId ncclId;
    if (rank == 0) {
      NCCL_CHECK(ncclGetUniqueId(&ncclId));
      std::cout << "Master generated NCCL Unique ID" << std::endl;
    }

    // 使用TCP/IP将NCCL ID广播给所有进程
    if (!TCPUtils::broadcastNCCLId(&ncclId, sizeof(ncclUniqueId), rank, worldSize, masterIP, port)) {
      throw std::runtime_error("Failed to broadcast NCCL ID");
    }

    // 初始化NCCL通信器
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, worldSize, ncclId, rank));
    std::cout << "Rank " << rank << " initialized NCCL communicator" << std::endl;

    // 创建CUDA流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 分配和初始化数据
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, dataSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, dataSize * sizeof(float)));

    // 给每个等级分配不同的值
    std::vector<float> h_data(dataSize, rank + 1.0f);  // 每个进程输入不同的值
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), dataSize * sizeof(float), cudaMemcpyHostToDevice));

    // 创建输出缓冲区
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, dataSize * sizeof(float)));

    // 同步所有进程
    NCCL_CHECK(ncclAllReduce(d_data, d_result, dataSize, ncclFloat, ncclSum, comm, stream));

    // 等待操作完成
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 检验结果
    std::vector<float> h_result(dataSize);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, dataSize * sizeof(float), cudaMemcpyDeviceToHost));

    // 计算预期结果：sum(1..worldSize)
    float expectedSum = worldSize * (worldSize + 1) / 2.0f;
    bool correct = true;

    // 检查结果（仅检查前几个元素）
    size_t checkLimit = std::min(dataSize, static_cast<size_t>(10));
    for (size_t i = 0; i < checkLimit; i++) {
      if (std::abs(h_result[i] - expectedSum) > 1e-5) {
        std::cout << "Rank " << rank << ": Validation failed at index " << i << ", got " << h_result[i] << ", expected "
                  << expectedSum << std::endl;
        correct = false;
        break;
      }
    }

    if (correct) {
      std::cout << "Rank " << rank << ": AllReduce validation passed!" << std::endl;
    }

    // 执行AllReduce基准测试
    std::cout << "Rank " << rank << ": Running AllReduce benchmark..." << std::endl;

    // 测量10次迭代的平均时间
    const int iterations = 10;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
      NCCL_CHECK(ncclAllReduce(d_data, d_result, dataSize, ncclFloat, ncclSum, comm, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    double avgTimeMs = (elapsed.count() * 1000) / iterations;

    // 计算带宽 (2*N: 发送+接收, *4: float是4字节)
    double bandwidth = (2.0 * dataSize * sizeof(float) * worldSize) / (avgTimeMs / 1000) / (1024 * 1024 * 1024);

    if (rank == 0) {
      std::cout << "AllReduce performance:" << std::endl
                << "  Data size: " << dataSize << " elements (" << (dataSize * sizeof(float) / (1024.0 * 1024.0))
                << " MB)" << std::endl
                << "  Average time: " << std::fixed << std::setprecision(3) << avgTimeMs << " ms" << std::endl
                << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
    }

    // 清理资源
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));

    std::cout << "Rank " << rank << " finished successfully" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
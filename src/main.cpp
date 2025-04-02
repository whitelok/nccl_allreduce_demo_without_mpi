#include <unistd.h>  // gethostname
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "gpu_worker.h"
#include "tcp_socket.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  try {
    // 解析命令行参数
    AppArgs args = AppArgs::parseArgs(argc, argv);
    args.printConfig();

    // 获取主机名
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    Logger::log("Running on host:", hostname, "as node", args.nodeRank);

    // 创建NCCL唯一ID
    ncclUniqueId ncclId;
    if (args.nodeRank == 0) {
      // 主节点生成NCCL ID
      NCCL_CHECK(ncclGetUniqueId(&ncclId));
      Logger::log("Generated NCCL Unique ID");
    }

    // 使用TCP/IP将NCCL ID广播到所有节点
    if (!NCCLIdBroadcaster::broadcastNCCLId(ncclId, args.nodeRank, args.nodeCount, args.masterIP, args.port)) {
      throw std::runtime_error("Failed to broadcast NCCL ID");
    }

    // 创建GPU工作线程管理器
    GPUWorkerManager gpuManager(args.nodeRank, args.nodeCount, args.dataSize, args.iterations);

    // 获取当前节点上的GPU数量
    int localDeviceCount = gpuManager.getLocalDeviceCount();
    Logger::log("Node", args.nodeRank, "managing", localDeviceCount, "GPUs");

    // 运行AllReduce操作
    if (!gpuManager.runAllreduce(ncclId)) {
      throw std::runtime_error("AllReduce operation failed");
    }

    // 输出性能结果
    double avgTime, avgBandwidth;
    gpuManager.getPerformanceStats(avgTime, avgBandwidth);

    Logger::log("\nNode", args.nodeRank, "Performance Summary:");
    Logger::log("  Data size:", args.dataSize, "elements (", (args.dataSize * sizeof(float) / (1024.0 * 1024.0)),
                "MB)");
    Logger::log("  Average time:", std::fixed, std::setprecision(3), avgTime, "ms");
    Logger::log("  Bandwidth:", std::fixed, std::setprecision(2), avgBandwidth, "GB/s");

    // 如果是主节点，等待一会儿让其他节点完成输出
    if (args.nodeRank == 0) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      Logger::log("\nAll operations completed successfully across", args.nodeCount, "nodes");
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
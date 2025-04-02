#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

class TCPSocket {
 public:
  // 构造和销毁
  TCPSocket();
  ~TCPSocket();

  // 禁止拷贝
  TCPSocket(const TCPSocket&) = delete;
  TCPSocket& operator=(const TCPSocket&) = delete;

  // 服务器功能
  bool listen(int port, int backlog = 10);
  std::unique_ptr<TCPSocket> accept();

  // 客户端功能
  bool connect(const std::string& host, int port);

  // 数据传输
  bool send(const void* data, size_t size);
  bool recv(void* buffer, size_t size);

  // 实用方法
  bool isConnected() const;
  void close();

  // 获取socket文件描述符
  int getSocketFD() const { return socketFD; }

 private:
  int socketFD;
  bool connected;
};

// 辅助函数, 用于简化多机器间通信
namespace TCPUtils {
// 在多机环境中交换NCCL ID
bool broadcastNCCLId(void* commId, size_t commIdSize, int rank, int worldSize, const std::string& masterIP, int port);
}  // namespace TCPUtils
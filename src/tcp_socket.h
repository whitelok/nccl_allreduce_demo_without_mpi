#pragma once

#include <memory>
#include <string>
#include <vector>

// TCP Socket类 - 用于节点间通信
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

 private:
  int socketFD;
  bool connected;
};

// 多节点NCCL ID交换功能
class NCCLIdBroadcaster {
 public:
  // 在多节点间广播NCCL ID
  static bool broadcastNCCLId(ncclUniqueId& ncclId, int nodeRank, int worldSize, const std::string& masterIP, int port);
};
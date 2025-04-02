#include "tcp_socket.h"
#include <cstring>
#include <iostream>

// 针对不同平台的网络库支持
#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
typedef int socklen_t;
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  define SOCKET_ERROR (-1)
#  define INVALID_SOCKET (-1)
typedef int SOCKET;
#endif

#include <chrono>
#include <thread>

// 调试输出宏
#ifdef DEBUG
#  define SOCKET_LOG(msg, ...) fprintf(stderr, "[TCP] " msg "\n", ##__VA_ARGS__)
#else
#  define SOCKET_LOG(msg, ...)
#endif

TCPSocket::TCPSocket() : socketFD(INVALID_SOCKET), connected(false) {
#ifdef _WIN32
  // 在Windows上初始化网络库
  static bool wsaInitialized = false;
  if (!wsaInitialized) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
      SOCKET_LOG("WSAStartup failed");
      return;
    }
    wsaInitialized = true;
  }
#endif

  socketFD = socket(AF_INET, SOCK_STREAM, 0);
  if (socketFD == INVALID_SOCKET) {
    SOCKET_LOG("Socket creation failed");
  }

  // 设置TCP_NODELAY以减少延迟
  int flag = 1;
  if (setsockopt(socketFD, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int)) == SOCKET_ERROR) {
    SOCKET_LOG("Set TCP_NODELAY failed");
  }
}

TCPSocket::~TCPSocket() { close(); }

bool TCPSocket::listen(int port, int backlog) {
  if (socketFD == INVALID_SOCKET) {
    SOCKET_LOG("Invalid socket in listen()");
    return false;
  }

  // 允许地址重用
  int opt = 1;
  if (setsockopt(socketFD, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt)) == SOCKET_ERROR) {
    SOCKET_LOG("setsockopt(SO_REUSEADDR) failed");
  }

  struct sockaddr_in serverAddr;
  memset(&serverAddr, 0, sizeof(serverAddr));
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  serverAddr.sin_port = htons(port);

  if (bind(socketFD, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
    SOCKET_LOG("Bind failed on port %d", port);
    return false;
  }

  if (::listen(socketFD, backlog) == SOCKET_ERROR) {
    SOCKET_LOG("Listen failed on port %d", port);
    return false;
  }

  SOCKET_LOG("Server listening on port %d", port);
  return true;
}

std::unique_ptr<TCPSocket> TCPSocket::accept() {
  if (socketFD == INVALID_SOCKET) {
    SOCKET_LOG("Invalid socket in accept()");
    return nullptr;
  }

  struct sockaddr_in clientAddr;
  socklen_t clientAddrLen = sizeof(clientAddr);
  SOCKET clientSocketFD = ::accept(socketFD, (struct sockaddr*)&clientAddr, &clientAddrLen);

  if (clientSocketFD == INVALID_SOCKET) {
    SOCKET_LOG("Accept failed");
    return nullptr;
  }

  std::unique_ptr<TCPSocket> clientSocket(new TCPSocket());
  clientSocket->socketFD = clientSocketFD;
  clientSocket->connected = true;

  char clientIP[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &(clientAddr.sin_addr), clientIP, INET_ADDRSTRLEN);
  SOCKET_LOG("Accepted connection from %s:%d", clientIP, ntohs(clientAddr.sin_port));

  return clientSocket;
}

bool TCPSocket::connect(const std::string& host, int port) {
  if (socketFD == INVALID_SOCKET) {
    SOCKET_LOG("Invalid socket in connect()");
    return false;
  }

  struct sockaddr_in serverAddr;
  memset(&serverAddr, 0, sizeof(serverAddr));
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(port);

  // 将主机名转换为IP地址
  if (inet_pton(AF_INET, host.c_str(), &serverAddr.sin_addr) <= 0) {
    // 尝试DNS解析
    struct addrinfo hints, *result = nullptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    int status = getaddrinfo(host.c_str(), nullptr, &hints, &result);
    if (status != 0) {
      SOCKET_LOG("Invalid address / Address not supported: %s", host.c_str());
      return false;
    }

    struct sockaddr_in* addr = (struct sockaddr_in*)result->ai_addr;
    serverAddr.sin_addr = addr->sin_addr;
    freeaddrinfo(result);
  }

  // 尝试连接，带重试
  const int maxRetries = 10;
  for (int retry = 0; retry < maxRetries; retry++) {
    if (::connect(socketFD, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) != SOCKET_ERROR) {
      connected = true;
      SOCKET_LOG("Connected to %s:%d", host.c_str(), port);
      return true;
    }

    SOCKET_LOG("Connection attempt %d/%d to %s:%d failed, retrying...", retry + 1, maxRetries, host.c_str(), port);

    // 重试前等待
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  SOCKET_LOG("Failed to connect to %s:%d after %d retries", host.c_str(), port, maxRetries);
  return false;
}

bool TCPSocket::send(const void* data, size_t size) {
  if (!isConnected()) {
    SOCKET_LOG("Not connected in send()");
    return false;
  }

  const char* buffer = static_cast<const char*>(data);
  size_t totalSent = 0;

  while (totalSent < size) {
    int sent = ::send(socketFD, buffer + totalSent, size - totalSent, 0);
    if (sent == SOCKET_ERROR) {
      SOCKET_LOG("Send failed");
      return false;
    }

    totalSent += sent;
  }

  return true;
}

bool TCPSocket::recv(void* buffer, size_t size) {
  if (!isConnected()) {
    SOCKET_LOG("Not connected in recv()");
    return false;
  }

  char* buf = static_cast<char*>(buffer);
  size_t totalReceived = 0;

  while (totalReceived < size) {
    int received = ::recv(socketFD, buf + totalReceived, size - totalReceived, 0);
    if (received <= 0) {
      if (received == 0) {
        SOCKET_LOG("Connection closed by peer");
      } else {
        SOCKET_LOG("Recv failed");
      }
      return false;
    }

    totalReceived += received;
  }

  return true;
}

bool TCPSocket::isConnected() const { return connected && socketFD != INVALID_SOCKET; }

void TCPSocket::close() {
  if (socketFD != INVALID_SOCKET) {
#ifdef _WIN32
    closesocket(socketFD);
#else
    ::close(socketFD);
#endif
    socketFD = INVALID_SOCKET;
  }
  connected = false;
}

// 实现TCP工具函数
namespace TCPUtils {

bool broadcastNCCLId(void* commId, size_t commIdSize, int rank, int worldSize, const std::string& masterIP, int port) {
  if (worldSize <= 0 || rank < 0 || rank >= worldSize) {
    SOCKET_LOG("Invalid rank (%d) or world size (%d)", rank, worldSize);
    return false;
  }

  // 如果是单机场景，不需要进行网络通信
  if (worldSize == 1) {
    return true;
  }

  if (rank == 0) {
    // 主进程：启动服务器并广播NCCL ID
    TCPSocket serverSocket;
    if (!serverSocket.listen(port)) {
      SOCKET_LOG("Master failed to listen on port %d", port);
      return false;
    }

    std::vector<std::unique_ptr<TCPSocket>> clientSockets;

    // 接受所有worker连接
    for (int i = 1; i < worldSize; i++) {
      auto clientSocket = serverSocket.accept();
      if (!clientSocket) {
        SOCKET_LOG("Failed to accept connection from worker %d", i);
        return false;
      }

      // 发送NCCL ID给worker
      if (!clientSocket->send(commId, commIdSize)) {
        SOCKET_LOG("Failed to send NCCL ID to worker %d", i);
        return false;
      }

      clientSockets.push_back(std::move(clientSocket));
    }

    SOCKET_LOG("Master successfully broadcast NCCL ID to all workers");
  } else {
    // Worker进程：连接到master并接收NCCL ID
    TCPSocket clientSocket;
    if (!clientSocket.connect(masterIP, port)) {
      SOCKET_LOG("Worker %d failed to connect to master at %s:%d", rank, masterIP.c_str(), port);
      return false;
    }

    // 接收NCCL ID
    if (!clientSocket.recv(commId, commIdSize)) {
      SOCKET_LOG("Worker %d failed to receive NCCL ID from master", rank);
      return false;
    }

    SOCKET_LOG("Worker %d successfully received NCCL ID from master", rank);
  }

  return true;
}

}  // namespace TCPUtils
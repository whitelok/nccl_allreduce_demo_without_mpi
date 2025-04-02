### NCCL多机多卡通讯样例（无MPI版）

每个机器一个进程，每个进程通过多线程管理这个机器上的多个GPU，一个线程管理一个GPU。

### 运行

```bash
# 创建并进入构建目录
mkdir -p build && cd build

# 配置CMake项目
cmake ..

# 编译
cmake --build .

#主节点上运行
./nccl_multi_node_demo --rank 0 --nproc 2 --port [通讯用的端口] --size [传输的数据大小]

#worker节点上运行
./nccl_multi_node_demo --rank 1 --nproc 2 --master [主节点IP] --port [通讯用的端口] --size [传输的数据大小]
```

### 主节点运行结果



### worker节点运行结果
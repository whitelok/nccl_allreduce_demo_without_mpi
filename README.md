```bash
# 创建并进入构建目录
mkdir -p build && cd build

# 配置CMake项目
cmake ..

# 编译
cmake --build .
```

主节点上运行

./nccl_multi_node_demo --rank 0 --nproc 2 --port 9999 --size 1000000

worker节点上运行

./nccl_multi_node_demo --rank 1 --nproc 2 --master [主节点IP] --port 9999 --size 1000000
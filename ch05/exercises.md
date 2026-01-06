# 第5章

### 练习 1

考虑矩阵加法。能否使用共享内存来减少全局内存带宽消耗?提示:分析每个线程访问的元素,看看线程之间是否有任何共同性。

### 练习 2

为 8 × 8 矩阵乘法绘制与图 5.7 等效的图,分别使用 2 × 2 分块和 4 × 4 分块。验证全局内存带宽的减少确实与分块的维度大小成正比。

### 练习 3

如果在图 5.9 的内核中忘记使用一个或两个 `__syncthreads()`,会发生什么类型的错误执行行为?

### 练习 4

假设寄存器或共享内存的容量不是问题,请给出一个重要原因,说明为什么使用共享内存而不是寄存器来保存从全局内存中获取的值是有价值的?解释你的答案。

### 练习 5

对于我们的分块矩阵-矩阵乘法内核,如果我们使用 32 × 32 的分块,输入矩阵 M 和 N 的内存带宽使用减少了多少?

### 练习 6

假设一个 CUDA 内核使用 1000 个线程块启动,每个线程块有 512 个线程。如果一个变量在内核中声明为局部变量,在内核执行的整个生命周期中将创建多少个该变量的版本?

### 练习 7

在上一个问题中,如果一个变量声明为共享内存变量,在内核执行的整个生命周期中将创建多少个该变量的版本?

### 练习 8

考虑对两个维度为 N × N 的输入矩阵执行矩阵乘法。在以下情况下,输入矩阵中的每个元素从全局内存请求多少次:

a. 没有分块?

b. 使用大小为 T × T 的分块?

### 练习 9

一个内核每个线程执行 36 次浮点运算和 7 次 32 位全局内存访问。对于以下每种设备属性,指出该内核是计算受限还是内存受限。

a. 峰值 FLOPS=200 GFLOPS,峰值内存带宽=100 GB/秒

b. 峰值 FLOPS=300 GFLOPS,峰值内存带宽=250 GB/秒

### 练习 10

为了操作分块,一位新手 CUDA 程序员编写了一个设备内核,该内核将转置矩阵中的每个分块。分块的大小为 BLOCK_WIDTH × BLOCK_WIDTH, 并且已知矩阵 A 的每个维度都是 BLOCK_WIDTH 的倍数。内核调用和代码如下所示。BLOCK_WIDTH 在编译时已知,可以设置为 1 到 20 之间的任何值。

```cpp
1  dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH);
2  dim3 gridDim(A_width/blockDim.x,A_height/blockDim.y);
3  BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

4  __global__ void
5  BlockTranspose(float* A_elements, int A_width, int A_height)
6  {
7      __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

8      int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
9      baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

10     blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

11     A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
12 }
```

a. 在 BLOCK_SIZE 的可能值范围内,对于哪些 BLOCK_SIZE 值,此内核函数将在设备上正确执行?

b. 如果代码不能对所有 BLOCK_SIZE 值正确执行,导致此错误执行行为的根本原因是什么?建议对代码进行修复,使其适用于所有 BLOCK_SIZE 值。

### 练习 11

考虑以下 CUDA 内核及调用它的相应主机函数:

```cpp
1  __global__ void foo_kernel(float* a, float* b) {
2      unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
3      float x[4];
4      __shared__ float y_s;
5      __shared__ float b_s[128];
6      for(unsigned int j = 0; j < 4; ++j) {
7          x[j] = a[j*blockDim.x*gridDim.x + i];
8      }
9      if(threadIdx.x == 0) {
10         y_s = 7.4f;
11     }
12     b_s[threadIdx.x] = b[i];
13     __syncthreads();
14     b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
15             + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
16 }
17 void foo(int* a_d, int* b_d) {
18     unsigned int N = 1024;
19     foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
20 }
```

a. 变量 i 有多少个版本?

b. 数组 x[] 有多少个版本?

c. 变量 y_s 有多少个版本?

d. 数组 b_s[] 有多少个版本?

e. 每个块使用的共享内存量是多少(以字节为单位)?

f. 内核的浮点运算与全局内存访问的比率是多少(以 OP/B 为单位)?

### 练习 12

考虑一个具有以下硬件限制的 GPU:每个 SM 2048 个线程、每个 SM 32 个块、每个 SM 64K(65,536)个寄存器,以及每个 SM 96 KB 共享内存。对于以下每种内核特性,指出该内核是否可以达到完全占用。如果不能,指出限制因素。

a. 内核使用每个块 64 个线程,每个线程 27 个寄存器,以及每个块 4 KB 共享内存。

b. 内核使用每个块 256 个线程,每个线程 31 个寄存器,以及每个块 8 KB 共享内存。

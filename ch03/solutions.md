# 第3章

### 解答 1

#### a

```c
// 每个kernel计算矩阵乘法的一行
__global__
void matrixRowKernel(float* M, float* N, float* P, int size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < size) {
    for (int col = 0; col < size; ++col) {
      float cur = 0;
      for (int j = 0; j < size; ++j) {
        cur += M[row * size + j] * N[j * size + col];
      }
      P[row * size + col] = cur;
    }
  }
}

```

#### b

```c
// 每个kernel计算矩阵乘法的一列
__global__
void matrixColKernel(float* M, float* N, float* P, int size) {
  // 只有这三行有区别
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < size) {
    for (int row = 0; row < size; ++row) {
    // 只有这三行有区别
      float cur = 0;
      for (int j = 0; j < size; ++j) {
        cur += M[row * size + i] * N[i * size + col];
      }
      P[row * size + col] = cur;
    }
  }
}
```

#### c

**共同缺点：**

1. **并行度利用不足**：两种方式都只用一个线程处理整行或整列，没有充分利用 GPU 的大规模并行能力
2. **计算效率低**：每个线程需要执行 O(n²) 次操作（n 为矩阵维度），而标准的每线程一元素方式只需 O(n) 次操作
3. **负载不均衡**：对于非方阵，两种方式都可能造成线程间的负载不均衡

如果M的行数大大超过列数，则计算行的方式更好（并行度更高，循环更少），反之，则计算列的方式更好。

### 解答 2

```c
// 矩阵-向量乘法: A[i] = sum(B[i][j] * C[j])
// 每个线程计算输出向量的一个元素
__global__
void matrixVecMul_kernel(float* B, float* C, float* A, int size) {
  // 只需要一维索引,因为输出是一个向量
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < size) {
    float sum = 0.0f;
    // 计算矩阵B的第row行与向量C的点积
    for (int j = 0; j < size; ++j) {
      sum += B[row * size + j] * C[j];
    }
    A[row] = sum;  // 输出是向量,不是矩阵
  }
}

void matrixVecMul(float* B_d, float* C_d, float* A_d, int size) {
  // 计算需要的块数(向上取整)
  int numBlocks = (size + 255) / 256;
  int threadsPerBlock = 256;

  // 一维grid和一维block配置
  matrixVecMul_kernel<<<numBlocks, threadsPerBlock>>>(B_d, C_d, A_d, size);
}
```

### 解答 3

#### a

每个块中有 16 \* 32 = 512 个线程

#### b

网格中共有 19 \* 5 \* 512 = 48640 个线程

#### c

网格中共有 19 \* 5 = 95 个块

#### d

总共执行的线程数是150 \* 300 = 45000 个线程

### 解答 4

#### a

行主序：20 \* 400 + 10 = 8010

#### b

列主序：10 \* 500 + 20 = 5020

### 解答 5

5 \* (400 \* 500) + 20 \* 400 + 10 = 1008010

# 第6章

### 解答 1

```cpp

#define TILE_WIDTH 16

__global__ void matrixMul_with_corner_turning_kernel(float *A, float *B,
                                                    float *P, int size) {

  __shared__ float Ad[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bd[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int col = bx * TILE_WIDTH + tx;
  int row = by * TILE_WIDTH + ty;

  float total = 0.0f;

  for (int ph = 0; ph < ceil(size / (float)TILE_WIDTH); ++ph) {
    if ((row < size) && (ph * TILE_WIDTH + tx) < size)
      Ad[ty][tx] = A[row * size + ph * TILE_WIDTH + tx];
    else
      Ad[ty][tx] = 0.0f;

    if ((ph * TILE_WIDTH + ty) < size && col < size)
      // 读取的时候在B数组中是顺序读取的
      // Bd[ty][0] = B[(ph * TILE_WIDTH + ty) * size + 0];
      // Bd[ty][1] = B[(ph * TILE_WIDTH + ty) * size + 1];
      Bd[ty][tx] = B[(ph * TILE_WIDTH + ty) * size + col];
    else
      Bd[ty][tx] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      // 共享内存中按Bd是按列读取的
      total += Ad[ty][k] * Bd[k][tx];
    }
    __syncthreads();
  }

  if ((row < size) && (col < size)) {
    P[row * size + col] = total;
  }
}
```

### 解答 2

> 合并访问需要一个warp里所有线程都能访问相邻的元素。

一个warp大小是32个线程。
如果block_size小于32，则一个warp中的线程会访问不同的共享内存，从而出现非合并访问。
所以block_size必须是32的倍数，比如32、64。

### 解答 3

a. 合并访问

b. 共享内存的访问，不涉及合并概念

c. 每次访问b的时候下标是 `j \* blockDim.x*gridDim.x + blockIdx.x \* blockDim.x + threadIdx.x`，
同一个块内只有threadIdx.x会变化，所以是合并访问。

d. 每次跳了4个i，所以不是合并访问

e. bc_s是共享内存，不涉及合并

f. a_s是共享内存，不涉及合并

g. 合并访问

h. 共享内存，不涉及合并

i. 每次跳了8个i，非合并访问

### 解答 4

- a.

```cpp
// CUDA核函数：矩阵乘法
__global__ void matrixMul_kernel(float *A, float *B, float *P, int size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < size && row < size) {
    float res = 0;
    for (int i = 0; i < size; ++i) {
      res += A[row * size + i] * B[i * size + col];
    }
    P[row * size + col] = res;
  }
}
```

浮点计算：size * 2

全局内存访问：size \*2\* 4B

比率：size \*2 / (size\* 2 \* 4B) = 0.25

- b.

```cpp
#define TILE_WIDTH 32

// CUDA核函数：矩阵乘法（带tiling）
__global__ void matrixMul_with_tiling_kernel(float *A, float *B, float *P,
                                             int size) {

  __shared__ float Ad[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bd[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int col = bx * TILE_WIDTH + tx;
  int row = by * TILE_WIDTH + ty;

  float total = 0;

  // ph代表phrase阶段，总共有size / TILE_WIDTH个阶段
  for (int ph = 0; ph < ceil(size / (float)TILE_WIDTH); ++ph) {
    if ((row < size) && (ph * TILE_WIDTH + tx) < size)
      Ad[ty][tx] = A[row * size + ph * TILE_WIDTH + tx];
    else
      Ad[ty][tx] = 0.0f;

    if ((ph * TILE_WIDTH + ty) < size && col < size)
      Bd[ty][tx] = B[(ph * TILE_WIDTH + ty) * size + col];
    else
      Bd[ty][tx] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      total += Ad[ty][k] * Bd[k][tx];
    }
    __syncthreads();
  }

  if ((row < size) && (col < size)) {
    P[row * size + col] = total;
  }
}
```

浮点计算：(size / 32) \* (1 + 32 \* 2) = 2.03 \* size

全局内存访问：(size / 32) \* (1 + 1) * 4B = 0.25 \* size

比率：2.03 / 0.25 = 8

- c.

```cpp
#define TILE_WIDTH 32
#define COARSE_FACTOR 4

// CUDA核函数：矩阵乘法（带线程加粗）
__global__ void matrixMul_with_thread_coarse_kernel(float *A, float *B,
                                                    float *P, int size) {

  __shared__ float Ad[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bd[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;
  int row = by * TILE_WIDTH + ty;

  float pv[COARSE_FACTOR];
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    pv[c] = 0.0f;
  }

  // ph代表phrase阶段，总共有size / TILE_WIDTH个阶段
  for (int ph = 0; ph < ceil(size / (float)TILE_WIDTH); ++ph) {
    if ((row < size) && (ph * TILE_WIDTH + tx) < size)
      Ad[ty][tx] = A[row * size + ph * TILE_WIDTH + tx];
    else
      Ad[ty][tx] = 0.0f;

    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int col = colStart + c * TILE_WIDTH;

      if ((ph * TILE_WIDTH + ty) < size && col < size)
        Bd[ty][tx] = B[(ph * TILE_WIDTH + ty) * size + col];
      else
        Bd[ty][tx] = 0.0f;

      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; ++k) {
        pv[c] += Ad[ty][k] * Bd[k][tx];
      }
      __syncthreads();
    }
  }

  for (int c = 0; c < COARSE_FACTOR; ++c) {
    int col = colStart + c * TILE_WIDTH;

    if ((row < size) && (col < size)) {
      P[row * size + col] = pv[c];
    }
  }
}

```

浮点计算：(size / 32) \* (1 + 4 \* 32 \* 2) = 8.03 \* size

全局内存访问：(size / 32) \* (1 + 4) \* 4B = 0.625 \* size

比率：8.03 / 0.625 = 12.85

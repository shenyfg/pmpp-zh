#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16
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

// 检查CUDA错误的辅助函数
#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__,              \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  // 默认矩阵大小
  int size = 512;

  // 检查命令行参数
  if (argc >= 2) {
    size = atoi(argv[1]);
    if (size <= 0 || size > 4096) {
      fprintf(stderr, "错误: 矩阵大小必须在 1-4096 之间\n");
      return 1;
    }
  }

  printf("矩阵乘法测试: %d x %d\n", size, size);

  // 计算矩阵大小
  const size_t bytes = size * size * sizeof(float);

  // 在主机上分配内存
  float *h_A = (float *)malloc(bytes);
  float *h_B = (float *)malloc(bytes);
  float *h_P = (float *)malloc(bytes);

  // 初始化矩阵 A 和 B
  printf("正在初始化矩阵...\n");
  for (int i = 0; i < size * size; i++) {
    h_A[i] = (float)(i % 10);       // A 矩阵: 0-9 循环
    h_B[i] = (float)((i % 10) * 2); // B 矩阵: 0-18 循环
  }

  // 在设备上分配内存
  float *d_A, *d_B, *d_P;
  CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&d_P, bytes));

  // 将矩阵数据从主机复制到设备
  CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  // 配置CUDA核函数启动参数
  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 blocksPerGrid((size + TILE_WIDTH - 1) / TILE_WIDTH,
                     (size + TILE_WIDTH - 1) / TILE_WIDTH);

  // 带线程加粗的配置（列方向的block数量减少COARSE_FACTOR倍）
  dim3 blocksPerGrid_coarse((size + TILE_WIDTH * COARSE_FACTOR - 1) /
                                (TILE_WIDTH * COARSE_FACTOR),
                            (size + TILE_WIDTH - 1) / TILE_WIDTH);

  printf("启动配置（不带线程加粗）: Grid(%d, %d), Block(%d, %d)\n",
         blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x,
         threadsPerBlock.y);
  printf("总线程数: %d\n", blocksPerGrid.x * blocksPerGrid.y *
                               threadsPerBlock.x * threadsPerBlock.y);

  printf("启动配置（带线程加粗）: Grid(%d, %d), Block(%d, %d)\n",
         blocksPerGrid_coarse.x, blocksPerGrid_coarse.y, threadsPerBlock.x,
         threadsPerBlock.y);
  printf("总线程数: %d, 线程加粗因子: %d\n",
         blocksPerGrid_coarse.x * blocksPerGrid_coarse.y * threadsPerBlock.x *
             threadsPerBlock.y,
         COARSE_FACTOR);

  // 创建CUDA事件用于计时
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  // 测试不带tiling的矩阵乘法
  printf("\n正在执行矩阵乘法（不带tiling）...\n");
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  matrixMul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_P, size);
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float milliseconds_no_tiling = 0;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds_no_tiling, start, stop));
  printf("不带tiling的执行时间: %.3f ms\n", milliseconds_no_tiling);

  // 测试带线程加粗的矩阵乘法
  printf("\n正在执行矩阵乘法（tiling + 线程加粗）...\n");
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  matrixMul_with_thread_coarse_kernel<<<blocksPerGrid_coarse,
                                        threadsPerBlock>>>(d_A, d_B, d_P, size);
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float milliseconds_with_coarse = 0;
  CHECK_CUDA_ERROR(
      cudaEventElapsedTime(&milliseconds_with_coarse, start, stop));
  printf("tiling + 线程加粗的执行时间: %.3f ms\n", milliseconds_with_coarse);

  // 计算加速比
  float speedup = milliseconds_no_tiling / milliseconds_with_coarse;
  printf("\n性能提升: %.2fx\n", speedup);
  printf("性能提升百分比: %.2f%%\n", (speedup - 1.0f) * 100.0f);

  // 清理事件
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  // 将结果从设备复制回主机
  CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost));

  printf("矩阵乘法完成!\n");

  // 验证结果: 对比几个采样点的CPU和GPU计算结果
  printf("\n验证结果（采样前10个元素）:\n");
  bool success = true;
  int checkCount = (size * size < 10) ? size * size : 10;

  for (int idx = 0; idx < checkCount; idx++) {
    int row = idx / size;
    int col = idx % size;

    // CPU 计算期望结果
    float expected = 0.0f;
    for (int i = 0; i < size; i++) {
      expected += h_A[row * size + i] * h_B[i * size + col];
    }

    float gpu_result = h_P[idx];
    printf("[%d,%d] GPU=%.2f, CPU=%.2f", row, col, gpu_result, expected);

    // 允许浮点误差
    if (fabs(gpu_result - expected) < 1e-3) {
      printf(" ✓\n");
    } else {
      printf(" ✗ (误差: %.2f)\n", fabs(gpu_result - expected));
      success = false;
    }
  }

  if (success) {
    printf("\n矩阵乘法测试通过!\n");
  } else {
    printf("\n矩阵乘法测试失败!\n");
  }

  // 清理内存
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_P);
  free(h_A);
  free(h_B);
  free(h_P);

  return 0;
}

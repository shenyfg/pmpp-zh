#include <cuda_runtime.h>
#include <stdio.h>

// CUDA核函数：简单的向量加法
__global__
void vectorAdd(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
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

int main() {
  const int N = 1024;
  const size_t size = N * sizeof(float);

  // 在主机上分配内存
  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);

  // 初始化输入数据
  for (int i = 0; i < N; i++) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  // 在设备上分配内存
  float *d_a, *d_b, *d_c;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));

  // 将数据从主机复制到设备
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

  // 启动CUDA核函数
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  printf("启动配置: <<<%d, %d>>>\n", blocksPerGrid, threadsPerBlock);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

  // 检查核函数执行是否有错误
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 将结果从设备复制回主机
  CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  // 打印前100个数
  for (int i = 0; i < 10; i++) {
    printf("h_c[%d] is %.2f\n", i, h_c[i]);
  }

  // 验证结果
  bool success = true;
  for (int i = 0; i < N; i++) {
    float expected = h_a[i] + h_b[i];
    if (fabs(h_c[i] - expected) > 1e-5) {
      printf("❌ 计算错误 index %d: expected %.2f, got %.2f\n", i, expected,
             h_c[i]);
      success = false;
      break;
    }
  }

  if (success) {
    printf("✓ 向量加法计算正确 (验证了前 %d 个元素)\n", N);
  }

  // 清理内存
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}

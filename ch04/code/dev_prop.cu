#include <cuda_runtime.h>
#include <stdio.h>

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
  printf("=== CUDA环境检测 ===\n\n");

  // 1. 检查CUDA设备数量
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess) {
    printf("❌ CUDA初始化失败: %s\n", cudaGetErrorString(error));
    printf("可能的原因:\n");
    printf("  - 未安装NVIDIA驱动\n");
    printf("  - 未安装CUDA Toolkit\n");
    printf("  - 没有支持CUDA的GPU\n");
    return 1;
  }

  if (deviceCount == 0) {
    printf("❌ 未检测到支持CUDA的GPU设备\n");
    return 1;
  }

  printf("✓ 检测到 %d 个CUDA设备\n\n", deviceCount);

  // 2. 输出每个设备的详细信息
  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));

    printf("设备 %d: %s\n", dev, deviceProp.name);
    printf("  计算能力: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  全局内存: %.2f GB\n",
           deviceProp.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("  多处理器数量: %d\n", deviceProp.multiProcessorCount);
    printf("  时钟频率: %.2f MHz\n", deviceProp.clockRate / 1000.0);
    printf("  内存时钟频率: %.2f MHz\n", deviceProp.memoryClockRate / 1000.0);
    printf("  内存带宽: %.2f GB/s\n", 2.0 * deviceProp.memoryClockRate *
                                          (deviceProp.memoryBusWidth / 8) /
                                          1.0e6);

    printf("  ✅块中允许的最大线程数: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  ✅设备中的SM数量: %d\n", deviceProp.multiProcessorCount);
    printf("  ✅块中每个纬度上的最大线程数，x: %d, y: %d, z: %d\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  ✅网格中每个维度的最大块数，x: %d, y: %d, z: %d\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  ✅每个SM允许的最大寄存器数: %d\n", deviceProp.regsPerBlock);
    printf("  ✅Warp大小: %d\n", deviceProp.warpSize);
    printf("  ✅每个SM中的共享内存量: %ld\n", deviceProp.sharedMemPerBlock);
    printf("\n");
  }

  return 0;
}

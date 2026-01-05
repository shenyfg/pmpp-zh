#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int CHANNELS = 3;

// CUDA核函数：彩色图像转灰度图像
__global__ void colortoGrayscaleConvertion(unsigned char *Pout,
                                           unsigned char *Pin, int width,
                                           int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * CHANNELS;
    unsigned char r = Pin[rgbOffset];
    unsigned char g = Pin[rgbOffset + 1];
    unsigned char b = Pin[rgbOffset + 2];
    Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
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
  // 检查命令行参数
  if (argc < 2) {
    printf("用法: %s <输入图片路径> [输出图片路径]\n", argv[0]);
    printf("示例: %s input.jpg output.png\n", argv[0]);
    return 1;
  }

  const char *inputPath = argv[1];
  const char *outputPath = (argc >= 3) ? argv[2] : "output_gray.png";

  // 读取图像
  int width, height, channels;
  printf("正在读取图像: %s\n", inputPath);
  unsigned char *h_colorImage =
      stbi_load(inputPath, &width, &height, &channels, CHANNELS);

  if (h_colorImage == NULL) {
    fprintf(stderr, "错误: 无法读取图像文件 %s\n", inputPath);
    return 1;
  }

  printf("图像信息: %d x %d, 通道数: %d\n", width, height, channels);

  // 计算图像大小
  const size_t colorImageSize =
      width * height * CHANNELS * sizeof(unsigned char);
  const size_t grayImageSize = width * height * sizeof(unsigned char);

  // 在主机上分配灰度图像内存
  unsigned char *h_grayImage = (unsigned char *)malloc(grayImageSize);

  // 在设备上分配内存
  unsigned char *d_colorImage, *d_grayImage;
  CHECK_CUDA_ERROR(cudaMalloc(&d_colorImage, colorImageSize));
  CHECK_CUDA_ERROR(cudaMalloc(&d_grayImage, grayImageSize));

  // 将彩色图像数据从主机复制到设备
  CHECK_CUDA_ERROR(cudaMemcpy(d_colorImage, h_colorImage, colorImageSize,
                              cudaMemcpyHostToDevice));

  // 配置CUDA核函数启动参数
  dim3 threadsPerBlock(16, 16); // 每个block 16x16=256个线程
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  printf("启动配置: Grid(%d, %d), Block(%d, %d)\n", blocksPerGrid.x,
         blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
  printf("总线程数: %d\n", blocksPerGrid.x * blocksPerGrid.y *
                               threadsPerBlock.x * threadsPerBlock.y);

  // 启动CUDA核函数
  colortoGrayscaleConvertion<<<blocksPerGrid, threadsPerBlock>>>(
      d_grayImage, d_colorImage, width, height);

  // 检查核函数执行是否有错误
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 将灰度图像结果从设备复制回主机
  CHECK_CUDA_ERROR(cudaMemcpy(h_grayImage, d_grayImage, grayImageSize,
                              cudaMemcpyDeviceToHost));

  printf("\n灰度转换完成!\n");

  // 保存灰度图像
  printf("正在保存灰度图像到: %s\n", outputPath);
  if (stbi_write_png(outputPath, width, height, 1, h_grayImage, width)) {
    printf("灰度图像保存成功: %s\n", outputPath);
  } else {
    fprintf(stderr, "错误: 无法保存图像文件 %s\n", outputPath);
  }

  // 清理内存
  cudaFree(d_colorImage);
  cudaFree(d_grayImage);
  stbi_image_free(h_colorImage);
  free(h_grayImage);

  return 0;
}

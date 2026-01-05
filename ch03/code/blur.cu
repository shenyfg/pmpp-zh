#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int BLUR_SIZE = 5;
const int CHANNELS = 3;

// CUDA核函数：图像模糊算法（支持RGB彩色图像）
__global__ void blur_kernel(unsigned char *Pout, unsigned char *Pin, int width,
                            int height, int channels) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    // 对每个颜色通道分别进行模糊处理
    for (int c = 0; c < channels; c++) {
      int pixVal = 0;
      int pixels = 0;

      // 在 BLUR_SIZE 范围内计算平均值
      for (int i = -BLUR_SIZE; i < BLUR_SIZE; ++i) {
        for (int j = -BLUR_SIZE; j < BLUR_SIZE; ++j) {
          int curRow = row + i;
          int curCol = col + j;

          // 边界检查
          if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
            int idx = (curRow * width + curCol) * channels + c;
            pixVal += Pin[idx];
            ++pixels;
          }
        }
      }

      // 计算当前像素的输出索引
      int outIdx = (row * width + col) * channels + c;
      Pout[outIdx] = (unsigned char)(pixVal / pixels);
    }
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
  const char *outputPath = (argc >= 3) ? argv[2] : "output_blur.png";

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

  // 计算图像大小（输入输出都是彩色图像）
  const size_t imageSize = width * height * CHANNELS * sizeof(unsigned char);

  // 在主机上分配输出图像内存
  unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);

  // 在设备上分配内存
  unsigned char *d_inputImage, *d_outputImage;
  CHECK_CUDA_ERROR(cudaMalloc(&d_inputImage, imageSize));
  CHECK_CUDA_ERROR(cudaMalloc(&d_outputImage, imageSize));

  // 将输入图像数据从主机复制到设备
  CHECK_CUDA_ERROR(cudaMemcpy(d_inputImage, h_colorImage, imageSize,
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
  blur_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_outputImage, d_inputImage,
                                                  width, height, CHANNELS);

  // 检查核函数执行是否有错误
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 将模糊后的图像结果从设备复制回主机
  CHECK_CUDA_ERROR(cudaMemcpy(h_outputImage, d_outputImage, imageSize,
                              cudaMemcpyDeviceToHost));

  printf("\n图像模糊处理完成!\n");

  // 保存模糊后的图像
  printf("正在保存模糊图像到: %s\n", outputPath);
  if (stbi_write_png(outputPath, width, height, CHANNELS, h_outputImage,
                     width * CHANNELS)) {
    printf("模糊图像保存成功: %s\n", outputPath);
  } else {
    fprintf(stderr, "错误: 无法保存图像文件 %s\n", outputPath);
  }

  // 清理内存
  cudaFree(d_inputImage);
  cudaFree(d_outputImage);
  stbi_image_free(h_colorImage);
  free(h_outputImage);

  return 0;
}

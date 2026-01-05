# CUDA图像处理程序 Makefile

# 编译器
NVCC = nvcc

# 编译选项
NVCCFLAGS = -O2 -arch=sm_50

# 目标可执行文件
TARGETS = color_to_gray blur matrix_mul

# 源文件
SRC_GRAY = color_to_gray.cu
SRC_BLUR = blur.cu
SRC_MATRIX = matrix_mul.cu

# 默认目标：编译所有程序
all: $(TARGETS)

# 编译灰度转换程序
color_to_gray: $(SRC_GRAY)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 编译模糊处理程序
blur: $(SRC_BLUR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 编译矩阵乘法程序
matrix_mul: $(SRC_MATRIX)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 运行灰度转换测试
run-gray: color_to_gray
	./color_to_gray

# 运行模糊处理测试
run-blur: blur
	./blur

# 运行矩阵乘法测试
run-matrix: matrix_mul
	./matrix_mul

# 清理
clean:
	rm -f $(TARGETS)

.PHONY: all run-gray run-blur run-matrix clean

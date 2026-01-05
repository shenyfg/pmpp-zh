# CUDA测试程序 Makefile

# 编译器
NVCC = nvcc

# 编译选项
NVCCFLAGS = -O2 -arch=sm_50

# 目标可执行文件
TARGET = dev_prop

# 源文件
SRC = dev_prop.cu

# 默认目标
all: $(TARGET)

# 编译规则
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 运行测试
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(TARGET)

.PHONY: all run clean

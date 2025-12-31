# 第2章

### 练习 1

如果我们想使用网格中的每个线程来计算向量加法的一个输出元素,那么将线程/块索引映射到数据索引(i)的表达式是什么?

**选项:**

- A. `i = threadIdx.x + threadIdx.y;`
- B. `i = blockIdx.x + threadIdx.x;`
- C. `i = blockIdx.x * blockDim.x + threadIdx.x;`
- D. `i = blockIdx.x * threadIdx.x;`

### 练习 2

假设我们想使用每个线程来计算向量加法的两个相邻元素。那么将线程/块索引映射到该线程要处理的第一个元素的数据索引(i)的表达式是什么?

**选项:**

- A. `i = blockIdx.x * blockDim.x + threadIdx.x * 2;`
- B. `i = blockIdx.x * threadIdx.x * 2;`
- C. `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`
- D. `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`
  
### 练习 3

我们想使用每个线程来计算向量加法的两个元素。每个线程块处理 2 * blockDim.x 个连续元素,这些元素形成两个部分。每个块中的所有线程将首先处理一个部分,每个线程处理一个元素。然后它们都将移动到下一个部分,每个线程处理一个元素。假设变量 i 应该是线程要处理的第一个元素的索引。那么将线程/块索引映射到第一个元素的数据索引的表达式是什么?

**选项:**

- A. `i=blockIdx.x*blockDim.x + threadIdx.x +2;`
- B. `i=blockIdx.x*threadIdx.x*2;`
- C. `i=(blockIdx.x*blockDim.x + threadIdx.x)*2;`
- D. `i=blockIdx.x*blockDim.x*2 + threadIdx.x;`

### 练习 4

对于向量加法,假设向量长度为 8000,每个线程计算一个输出元素,线程块大小为 1024 个线程。程序员配置内核调用使用最少数量的线程块来覆盖所有输出元素。网格中将有多少个线程?

**选项:**

- A. 8000
- B. 8196
- C. 8192
- D. 8200  

### 练习 5

如果我们想在 CUDA 设备全局内存中分配一个包含 v 个整数元素的数组,那么 `cudaMalloc` 调用的第二个参数的合适表达式是什么?

**选项:**

- A. n
- B. v
- C. n * sizeof(int)
- D. v * sizeof(int)

### 练习 6

如果我们想分配一个包含 n 个浮点数元素的数组,并让浮点指针变量 A_d 指向分配的内存,那么 `cudaMalloc` 调用的第一个参数的合适表达式是什么?

**选项:**

- A. n
- B. (void*) A_d
- C. *A_d
- D. (void**) &A_d

### 练习 7

如果我们想从主机数组 A_h(A_h 是源数组的元素 0 的指针)复制 3000 字节的数据到设备数组 A_d(A_d 是目标数组的元素 0 的指针),那么在 CUDA 中进行此数据复制的合适 API 调用是什么?

**选项:**

- A. cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);
- B. cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);
- C. cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);
- D. cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);

### 练习 8

如何声明一个变量 err 来适当地接收 CUDA API 调用的返回值?

**选项:**

- A. int err;
- B. cudaError err;
- C. cudaError_t err;
- D. cudaSuccess_t err;

### 练习 9

考虑以下 CUDA 内核及其对应的调用它的主机函数:

```c
01 __global__ void foo_kernel(float* a, float* b, unsigned int N) {
02     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
03
04     if (i < N) {
05         b[i] = 2.7f * a[i] - 4.3f;
06     }
07 }
08
09 void foo(float* a_d, float* b_d) {
10     unsigned int N = 200000;
11     foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
12 }
```

**a. 每个块中有多少个线程?**

**b. 网格中有多少个线程?**

**c. 网格中有多少个块?**

**d. 有多少个线程执行第 02 行的代码?**

**e. 有多少个线程执行第 04 行的代码?**

### 练习 10

一位新来的暑期实习生对 CUDA 感到沮丧。他一直在抱怨 CUDA 非常繁琐。他必须将计划在主机和设备上执行的许多函数声明两次,一次作为主机函数,一次作为设备函数。你的回应是什么?

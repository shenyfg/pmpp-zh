# 第6章

### 练习 1

编写一个矩阵乘法内核函数，对应于图6.4中所示的设计。

### 练习 2

对于分块矩阵乘法，在BLOCK_SIZE的可能取值范围内，对于哪些BLOCK_SIZE的值，内核可以完全避免对全局内存的非合并访问？（你只需要考虑方形块。）

### 练习 3

考虑以下CUDA内核：

```cpp
01  __global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
02  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
03  __shared__ float a_s[256];
04  __shared__ float bc_s[4*256];
05  a_s[threadIdx.x] = a[i];
06  for(unsigned int j = 0; j < 4; ++j) {
07      bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];
08  }
09  __syncthreads();
10  d[i + 8] = a_s[threadIdx.x];
11  e[i*8] = bc_s[threadIdx.x*4];
12  }
```

对于以下每个内存访问，请指明它们是合并访问、非合并访问还是不适用合并概念：

- a. 第05行对数组a的访问
- b. 第05行对数组a_s的访问
- c. 第07行对数组b的访问
- d. 第07行对数组c的访问
- e. 第07行对数组bc_s的访问
- f. 第10行对数组a_s的访问
- g. 第10行对数组d的访问
- h. 第11行对数组bc_s的访问
- i. 第11行对数组e的访问

### 练习 4

以下每个矩阵-矩阵乘法内核的浮点运算与全局内存访问比率（单位：OP/B）是多少？

假设我们有一个大小为(m, n)的矩阵M和一个大小为(n, o)的矩阵N。假设数据类型为float32，即4字节。

a. 第3章"多维网格和数据"中描述的简单内核，没有应用任何优化。

b. 第5章"内存架构和数据局部性"中描述的内核，使用32×32的分块大小应用了共享内存分块技术。

c. 本章中描述的内核，使用32×32的分块大小应用了共享内存分块技术，并使用加粗因子4应用了线程加粗技术。

# Notes on _CUDA Fortran for scientists and engineers_

> Ruetsch, G., & Fatica, M. (2013). CUDA Fortran for scientists and engineers: best practices for efficient CUDA Fortran programming. Elsevier.

## Chapter 1. Introduction

Principle reason for the ubiquitous presence of **multiple cores** CPUs - **inability** of CPU manufacturers to increase performance in single-core designs by **boosting the clock speed**.

- GPU - fine-grained model (data parallelism)
- MPI - coarse-grained model

### Synchronization

- Data transfer via **assignment statements** are blocking or synchronous transfers! (implicit synchronization)
- **Kernel launches** are **nonblocking or asynchronous**!

### Grouping of threads

- hardware: thread processors - grouped into **multiprocessors** (caontain a `shared memory')
- software: threads - grouped into **thread blocks** - kernel launches a **grid** of thread blocks
- `threadIdx` - the index of a thread within its thread block
- `blockDim` - the number of threads in a block
- `blockIdx` - the index of the block within the grid

### Hardware features

**Compute Capability**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

### Error handling

- synchronous error:
  ```
  ierrSync = cudaGetLastError()
  ```
- asynchronous error:
  ```
  ierrAsync = cudaDeviceSynchronize()
  ```

### Compilation

```
pgf90 ch1_1.cuf
pgf90 -Mcuda ch1_1.f90
```

- separate compilation: `-Mcuda=rdc` _relocatable device code_

## Chapter 2. Performance Measurement and Metrics

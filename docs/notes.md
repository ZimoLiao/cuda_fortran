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

### Measuring kernel execution time

- Care must be taken: many CUDA Fortran API functions are nonblocking or asynchronous

- to accurately time the kernel execution with host code timers, we need to **explicitly synchronize** the CPU thread using
  `cudaDeviceSynchronize()`

#### Timing via CUDA events

- **CUDA events** make use of the concept of _CUDA streams_ - all operations on the GPU have occurred in the default stream 0

```
  istat = cudaEventCreate(startEvent)
  istat = cudaEventCreate(stopEvent)

  istat = cudaEventRecord(startEvent, 0)
  call increment<<<ceiling(real(n)/tPB),tPB>>>(a_d, b)
  istat = cudaEventRecord(stopEvent, 0)
  istat = cudaEventSynchronize(stopEvent)
  istat = cudaEventElapsedTime(time, startEvent, stopEvent)

  istat = cudaEventDestroy(startEvent)
  istat = cudaEventDestroy(stopEvent)
```

<!-- #### Command line profiler (DEPRECATED) -->

#### The `nvprof` profiling tool

https://docs.nvidia.com/cuda/profiler-users-guide/index.html

```
nvprof ./ApplicationName
```

- `--print-gpu-trace` separate output for each call

### Instruction, bandwidth, and latency bound kernels

- **memory-intensive** or **math-intensive**

### Memory bandwidth

Bandwidth - _the rate at which data can be transferred_ - is one of the most important gating factors for performance

- the choice of memory in which data are stored
- how the data are laid out
- the order in which they are accessed

**Theoretical peak bandwidth** can be calculated from the memory clock and the memory bus width, quantities can be queried through `cudaGetDeviceProperties()`

**Effective bandwidth** is calculated by timing specific program activities and by knowing how data are accessed by the program

$$
BW_{effective}=\frac{(R_B+W_B)/10^9}{t}\quad(\mathrm{GB/s})
$$

where $R_B$ is the number of bytes read per kernel, $W_B$ is the number of bytes written per kernel, $t$ is the elapsed time given in seconds.

## Chapter 3. Optimization

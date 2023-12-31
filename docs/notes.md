# Notes on _CUDA Fortran for scientists and engineers_

> Ruetsch, G., & Fatica, M. (2013). CUDA Fortran for scientists and engineers: best practices for efficient CUDA Fortran programming. Elsevier.

[toc]

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

#### Command line profiler (DEPRECATED)

#### The `nvprof` profiling tool (DEPRECATED)

<!-- https://docs.nvidia.com/cuda/profiler-users-guide/index.html

```
nvprof ./ApplicationName
```

- `--print-gpu-trace` separate output for each call -->

#### Nsight system

https://developer.nvidia.com/nsight-systems

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

Two categories of data transfers:

1. data transfer between host and device memories
2. data transfer between different memories on the device

**Avoid transfers between the host and device whenever possible!**

### 3.1 Transfer between host and device

**_考虑 managed memory!!_**
<!-- https://developer.nvidia.com/blog/unified-memory-cuda-beginners/ -->
https://developer.nvidia.com/blog/unified-memory-cuda-fortran-programmers/

#### Pinned memory

Page table: https://en.wikipedia.org/wiki/Page_table

- The cost of the transfer between pageable memory and pinned host buffer can be avoided if we declare the host arrays to use pinned memory

- Pinned memory should **not be overused**, since excessive use can reduce overall system performance.

#### Batching small data transfers

transfer array sections between device and host:
use `cudaMemcpy2D()` or `cudaMemcpy3D()`

#### Asynchronous data transfers

### 3.2 Device memory

Types:

- In device DRAM:
  - **Global memory** - declared with `device` attribute in host code, can be read and written from both host and device
  - **Local memory**
  - **Constant memory** - declared using `constant` qualifier in a Fortran module. can be read and written from host but is read-only from threads in device. **constant data is cached on the chip and is most effective when threads that execute at the same time access the same value**
  - **Texture memory** - similar to constant memory
- On-chip:
  - **Shared memory** - accessible by all threads in a thread block. declared in device code using `shared` qualifier

![Device memory](memory.png)

#### coalesced access to global memory

**warp** (32 threads) - the actual grouping of threads that gets calculated in singel-instruction multiple-thread (SIMT) fashion.

- grouping into warps is relevant not only to computation but also to _global memory accesses_

#### texture memory

#### local memory

_local_ refers to a variable's scope (meaning **thread-private**) and not to its physical location (off-chip in device DRAM)

- **register memory is not indexable!**

#### constant memory

64KB of constant memory, cached on-chip

### 3.3 On-chip memory

#### L1 cache

#### Registers

#### Shared memory

shared memory is allocated per thread block

`syncthreads()` - barrier synchronization

- shared memory is divided into equally sized memory modules (banks) that can be accessed simultaneously.
- if multiple addresses of a memory request map to the same memory bank, the accesses are serialized.
- multiple accesses to the same location by any number of threads within a warp are served simultaneously.

### 3.4 Memory optimization example: matrix transpose

- memory coalescence (cache blcoking scheme)
- **avoid bank conflict** https://developer.download.nvidia.com/CUDA/training/sharedmemoryusage_july2011.mp4

### 3.5 Execution configuration

#### Thread-level parallelism

> 就是如何组织 block 和 grid 来调用核函数以达到最优

**Occupancy** - (used to help assess the thread-level parallelism of a kernel on a multiprocessor) - _the ratio of the number of active warps per multiprocessor to the maximum number of possible active warps_

- higher occupancy does not imply better performance

#### Instruction-level parallelism

### 3.6 Instruction optimization

#### Device intrinsics

- fast math intrinsics

### 3.7 Kernel loop directives (CUF kernels)

- These directives instruct the compiler to generate kernels from a region of host code consisting of tightly nested loops.

```
!$cuf kernel do <<<*,*>>>                 ! 1D
!$cuf kernel do <<<*,*,0,streamID>>>      ! 1D
!$cuf kernel do <<<*,*,stream=streamID>>> ! 1D

!$cuf kernel do (2) <<<(*,*),(32,8)>>>    ! 2D
!$cuf kernel do (2) <<<(*,*),(*,*)>>>     ! 2D
!$cuf kernel do (2) <<<*,*>>>             ! 2D
```

#### Reductions in CUF kernels! (自动优化实现)

#### Intruction-level parallelism

## Chapter 4. Multi-GPU Programming

- CUDA is compatible with any host threading model, such as OpenMP and MPI
- each host thread can access either single or multiple GPUs

### 4.1 CUDA multi-GPU features

- All CUDA calls are issued to the _current_ GPU, and `cudaSetDevice()` sets the current GPU
- Device arrays that are not declared with the `allocatable` attribute are implicitly allocated on the default device (device 0)

### 4.2 Multi-GPU Programming with MPI

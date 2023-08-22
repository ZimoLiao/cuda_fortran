module kernel_m
  implicit none

contains

  attributes(global) subroutine base(a, b)
  real :: a(*), b(*)
  integer :: i
  i = blockDim%x*(blockIdx%x-1) + threadIdx%x
  a(i) = sin(b(i))
end subroutine base

attributes(global) subroutine memory(a, b)
real :: a(*), b(*)
integer :: i
i = blockDim%x*(blockIdx%x-1) + threadIdx%x
a(i) = b(i)
end subroutine memory

attributes(global) subroutine math(a, b, flag)
real :: a(*)
real, value :: b
integer, value :: flag
real :: v
integer :: i
i = blockDim%x*(blockIdx%x-1) + threadIdx%x
v = sin(b)
if ( v*flag == 1 ) then
  a(i) = v
end if
end subroutine math

end module kernel_m

program limitingFactor
  use cudafor
  use kernel_m
  implicit none

  integer, parameter :: n = 8*1024*1024, blockSize = 256
  real, managed :: a(n), b(n)

  b = 1.0
  call base<<<n/blockSize,blockSize>>>(a, b)
  call memory<<<n/blockSize,blockSize>>>(a, b)
  call math<<<n/blockSize,blockSize>>>(a, 1.0, 0)

  write(*,*) a(1)

end program limitingFactor


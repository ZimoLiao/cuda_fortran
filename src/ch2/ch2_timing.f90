module simpleOps_m
  implicit none

contains

  attributes(global) subroutine increment(a, b)
  implicit none
  integer, intent(inout) :: a(:)
  integer, value :: b
  integer :: i, n

  i = blockDim%x*(blockIdx%x-1) + threadIdx%x
  n = size(a)
  if ( i<=n ) then
    a(i) = a(i)+b
  end if

end subroutine increment

end module simpleOps_m


program incrementTestGPU
  use cudafor
  use simpleOps_m
  implicit none

  integer, parameter :: n = 1024*1024;
  integer, allocatable :: a(:)
  integer :: b
  integer, device, allocatable :: a_d(:)
  type(cudaEvent) :: startEvent, stopEvent
  real :: time
  integer :: istat
  integer :: tPB = 256;

  allocate(a(n), a_d(n))
  a = 1
  b = 3

  istat = cudaEventCreate(startEvent)
  istat = cudaEventCreate(stopEvent)

  a_d = a
  istat = cudaEventRecord(startEvent, 0)
  call increment<<<ceiling(real(n)/tPB),tPB>>>(a_d, b)
  istat = cudaEventRecord(stopEvent, 0)
  istat = cudaEventSynchronize(stopEvent)
  istat = cudaEventElapsedTime(time, startEvent, stopEvent)
  a = a_d

  if ( any(a/=4) ) then
    write(*,*) '**** Program Failed ****'
  else
    write(*,*) 'Program Passed. Time for kernel execution (ms): ', time
  end if

  deallocate(a, a_d)
  istat = cudaEventDestroy(startEvent)
  istat = cudaEventDestroy(stopEvent)

end program incrementTestGPU


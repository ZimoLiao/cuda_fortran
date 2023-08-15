module simpleOps_m
  implicit none

contains

  attributes(global) subroutine increment(a, b)
  implicit none
  integer, intent(inout) :: a(:)
  integer, value :: b
  integer :: i

  i = threadIdx%x
  a(i) = a(i)+b

end subroutine increment

end module simpleOps_m


program incrementTestGPU
  use cudafor
  use simpleOps_m
  implicit none

  integer, parameter :: n = 256;
  integer :: a(n), b
  integer, device :: a_d(n)

  a = 1
  b = 3

  a_d = a
  call increment<<<1,n>>>(a_d, b)
  a = a_d

  if ( any(a/=4) ) then
    write(*,*) '**** Program Failed ****'
  else
    write(*,*) 'Program Passed'
  end if

end program incrementTestGPU


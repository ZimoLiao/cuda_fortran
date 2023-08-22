program peakBandwidth
  use cudafor
  implicit none

  integer :: i, istat, nDevices=0
  type (cudaDeviceProp) :: prop

  istat = cudaGetDeviceCount(nDevices)
  do i = 0, nDevices-1
    istat = cudaGetDeviceProperties(prop, i)
    write(*,"(' Device Number: ',i0)") i
    write(*,"('   Device name: ',a)") trim(prop%name)
    write(*,"('   Memory Clock Rate (KHz): ', i0)") &
      prop%memoryClockRate
    write(*,"('   Memory Bus Width (bits): ', i0)") &
      prop%memoryBusWidth
    write(*,"('   Peak Memory Bandwidth (GB/s): ', f6.2)") &
      2.0 * prop%memoryClockRate * &
      (prop%memoryBusWidth / 8) * 1.e-6
    write(*,*)
  enddo
end program peakBandwidth

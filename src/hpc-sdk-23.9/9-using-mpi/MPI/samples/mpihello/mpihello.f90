program hello
use mpi
integer ierr, myproc, nerrors
nerrors = 0
call mpi_init(ierr)
if (ierr .ne. 0) then
   nerrors = nerrors + 1
endif
call mpi_comm_rank(MPI_COMM_WORLD, myproc, ierr)
if (ierr .ne. 0) then
   nerrors = nerrors + 1
endif

print *, "Hello world!  I'm node", myproc

call mpi_finalize(ierr)
if (ierr .ne. 0) then
   nerrors = nerrors + 1
endif
if (nerrors .ne. 0) then
   print *, "Test FAILED"
else
   print *, "Test PASSED"
endif
end

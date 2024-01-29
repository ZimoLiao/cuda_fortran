#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"
extern int __NUMNODES, __MYPROCID;static MPI_Status _mpi_status;static int _n, _MPILEN;

/*
   Test of single process memcpy.
   ctx is ignored for this test.
*/
double memcpy_rate(reps,len,ctx)
int      reps,len;
void     *ctx;
{
  double elapsed_time;
  int  i,msg_id,myproc;
  char *sbuffer,*rbuffer;
  double t0, t1;

  sbuffer = (char *)malloc(len);
  rbuffer = (char *)malloc(len);

  myproc       = __MYPROCID;
  elapsed_time = 0;
  *(&t0)=MPI_Wtime();
  for(i=0;i<reps;i++){
      memcpy( rbuffer, sbuffer, len );
  }
  *(&t1)=MPI_Wtime();
  elapsed_time = *(&t1 )-*(&t0);

  free(sbuffer);
  free(rbuffer);
  return(elapsed_time);
}

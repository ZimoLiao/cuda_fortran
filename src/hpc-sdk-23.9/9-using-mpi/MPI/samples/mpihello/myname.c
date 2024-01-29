#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <stdio.h>
#include <Winsock2.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <unistd.h>
#endif
#include "mpi.h"
main(int argc, char **argv)
{
	int len, ierr;
	char hname[32];
	len = 32;
	MPI_Init(&argc, &argv);
	gethostname(hname, len);
	printf("My name is %s\n", hname);
	MPI_Finalize();
}

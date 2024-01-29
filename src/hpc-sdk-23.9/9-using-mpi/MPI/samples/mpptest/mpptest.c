/* This code was generated automatically from an older Chameleon program */

int __NUMNODES, __MYPROCID  ;




#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"
extern int __NUMNODES, __MYPROCID;static MPI_Status _mpi_status;static int _n, _MPILEN;


#ifndef DEFAULT_REPS
#define DEFAULT_REPS 50
#endif

/*
    This is a simple program to test the communications performance of
    a parallel machine.
 */

void *PairInit();
void *BisectInit();
void PairChange();
void BisectChange();
void *GOPInit();
void *OverlapInit();

double (*GetPairFunction())();
double (*GetGOPFunction())();
double memcpy_rate();
void RunATest();

/* Overlap testing */
double round_trip_nb_overlap();
double round_trip_b_overlap();

/* Routine to generate graphics context */
void *SetupGraph();

/* Prototypes */
double RunSingleTest();
void time_function();

/* These statics (globals) are used to estimate the parameters in the
   basic (s + rn) complexity model
   Sum of lengths is stored as a double to give 53 bit integer on
   systems where sizeof(int) == 4.
 */
static double sumtime = 0.0, sumlentime = 0.0;
static double sumlen  = 0.0,  sumlen2 = 0.0;
static double sumtime2 = 0.0;
static int    ntest   = 0;

/* If doinfo is 0, don't write out the various text lines */
static int    doinfo = 1;

/* Scaling of time and rate */
static double TimeScale = 1.0;
static double RateScale = 1.0;

/* This is the number of times to run a test, taking as time the minimum
   achieve timing.  This uses an adaptive approach that also stops when
   minThreshTest values are within a few percent of the current minimum */
static int    minreps       = 30;
static int    minThreshTest = 3;
static double repsThresh    = 0.05;
static int    NatThresh     = 3;
char   protocol_name[256];
/*
   We would also like to adaptively modify the number of repetitions to
   meet a time estimate (later, we'd like to meet a statistical estimate).

   One relatively easy way to do this is to use a linear estimate (either
   extrapolation or interpolation) based on 2 other computations.
   That is, if the goal time is T and the measured tuples (time,reps,len)
   are, the formula for the local time is s + r n, where

   r = (time2/reps2 - time1/reps1) / (len2 - len1)
   s = time1/reps1 - r * len1

   Then the appropriate number of repititions to use is

   Tgoal / (s + r * len) = reps
 */
static double Tgoal = 1.0;
/* If less than Tgoalmin is spent, increase the number of repititions */
static double TgoalMin = 0.5;
static int    AutoReps = 0;

/* This structure allows a collection of arbitray sizes to be specified */
#define MAX_SIZE_LIST 256
static int sizelist[MAX_SIZE_LIST];
static int nsizes = 0;

/* We wish to control the TOTAL amount of time that the test takes.
   We could do this with gettimeofday or clock or something, but fortunately
   the MPI timer is an elapsed timer */
static double max_run_time = 15.0*60.0;
static double start_time = 0.0;

/* These are used to contain results for a single test */
typedef struct {
    double len, t, mean_time, rate;
    double max_time,        /* max of the observations */
           smean_time;      /* smean is the mean of the observations */
    int    reps;
    } TwinResults;
typedef struct {
    double (*f)();
    int    reps, proc1, proc2;
    void   *msgctx;
    /* Here is where we should put "recent" timing data used to estimate
       the values of reps */
    double t1, t2;
    int    len1, len2;
    } TwinTest;

/*
   This function manages running a test and putting the data into the
   accumulators.  The information is placed in result.

   This function is intended for use by TSTAuto1d.  That routine controls
   the parameters passed to this routine (the value of x) and accumulates
   the results based on parameters (for accuracy, completeness, and
   "smoothness") passed to the TST routine.
 */
double GeneralF( x, result, ctx )
double      x;
TwinResults *result;
TwinTest    *ctx;
{
    double t, mean_time;
    int    len = (int)x, k;
    int    reps = ctx->reps;
    int    flag, iwork;
    double tmax, tmean;
    int    rank;

    if (AutoReps) {
	reps = GetRepititions( ctx->t1, ctx->t2, ctx->len1, ctx->len2,
			       len, reps );
    }

    t = RunSingleTest( ctx->f, reps, len, ctx->msgctx, &tmax, &tmean );

    mean_time	   = t / reps;              /* take average over trials */
    result->t	   = t;
    result->len	   = x;
    result->reps	   = reps;
    result->mean_time  = mean_time;
    result->smean_time = tmean;
    result->max_time   = tmax;
    if (mean_time > 0.0)
	result->rate      = ((double)len) / mean_time;
    else
	result->rate      = 0.0;

    /* Save the most recent timing data */
    ctx->t1     = ctx->t2;
    ctx->len1   = ctx->len2;
    ctx->t2     = mean_time;
    ctx->len2   = len;

    sumlen     += len;
    sumtime    += mean_time;
    sumlen2    += ((double)len) * ((double)len);
    sumlentime += mean_time * len;
    sumtime2   += mean_time * mean_time;
    ntest      ++;

    /* We need to insure that everyone gets the same result */
    MPI_Bcast(&result->rate, sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD );

    /* Check for max time exceeded */
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if (rank == 0 && MPI_Wtime() - start_time > max_run_time) {
	fprintf( stderr, "Exceeded %f seconds, aborting\n", max_run_time );
	MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    return result->rate;
}

int main(argc,argv)
int argc;
char *argv[];
{
    int    dist;
    double (* f)();
    void *MsgCtx = 0; /* This is the context of the message-passing operation */
    void *outctx;
    void (*ChangeDist)() = 0;
    int  reps,proc1,proc2,len,error_flag,distance_flag,distance;
    double t;
    int  first,last,incr, svals[3];
    int      autosize = 0, autodx;
    double   autorel  = 0.02;
    char     units[32];         /* Name of units of length */

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &__NUMNODES );
    MPI_Comm_rank( MPI_COMM_WORLD, &__MYPROCID );
    ;
    strcpy( protocol_name, "blocking" );
    strcpy( units, "(bytes)" );

    if (SYArgHasName( &argc, argv, 1, "-help" )) {
	return PrintHelp( argv );
    }

    if (__NUMNODES < 2) {
	fprintf( stderr, "Must run mpptest with at least 2 nodes\n" );
	return 1;
    }

/* Get the output context */
    outctx = SetupGraph( &argc, argv );
    if (SYArgHasName( &argc, argv, 1, "-noinfo" ))    doinfo    = 0;

    reps          = DEFAULT_REPS;
    proc1         = 0;
    proc2         = __NUMNODES-1;
    error_flag    = 0;
    distance_flag = 0;
    svals[0]      = 0;
    svals[1]      = 1024;
    svals[2]      = 32;

    if (SYArgHasName( &argc, argv, 1, "-distance" ))  distance_flag++;
    SYArgGetIntVec( &argc, argv, 1, "-size", 3, svals );
    nsizes = SYArgGetIntList( &argc, argv, 1, "-sizelist", MAX_SIZE_LIST,
			      sizelist );

    SYArgGetInt(    &argc, argv, 1, "-reps", &reps );
    if (SYArgHasName( &argc, argv, 1, "-autoreps" ))  AutoReps  = 1;
    if (SYArgGetDouble( &argc, argv, 1, "-tgoal", &Tgoal )) {
	AutoReps = 1;
	if (TgoalMin > 0.1 * Tgoal) TgoalMin = 0.1 * Tgoal;
    }
    SYArgGetDouble( &argc, argv, 1, "-rthresh", &repsThresh );

    SYArgGetInt( &argc, argv, 1, "-sample_reps", &minreps );

    autosize = SYArgHasName( &argc, argv, 1, "-auto" );
    if (autosize) {
	autodx = 4;
	SYArgGetInt( &argc, argv, 1, "-autodx", &autodx );
	autorel = 0.02;
	SYArgGetDouble( &argc, argv, 1, "-autorel", &autorel );
    }

/* Pick the general test based on the presence of an -gop, -overlap, -bisect
   or no arg */
    SetPattern( &argc, argv );
    if (SYArgHasName( &argc, argv, 1, "-gop")) {
	f      = GetGOPFunction( &argc, argv, protocol_name, units );
	MsgCtx = GOPInit( &argc, argv );
    }
    else if (SYArgHasName( &argc, argv, 1, "-bisect" )) {
	f = GetPairFunction( &argc, argv, protocol_name );
	dist = 1;
	SYArgGetInt( &argc, argv, 1, "-bisectdist", &dist );
	MsgCtx     = BisectInit( dist );
	ChangeDist = BisectChange;
	strcat( protocol_name, "-bisect" );
	if (SYArgHasName( &argc, argv, 1, "-debug" ))
	    PrintPairInfo( MsgCtx );
	TimeScale = 0.5;
	RateScale = (double) __NUMNODES; /* * (2 * 0.5) */
    }
    else if (SYArgHasName( &argc, argv, 1, "-overlap" )) {
	int MsgSize;
	char cbuf[32];
	if (SYArgHasName( &argc, argv, 1, "-sync" )) {
	    f = round_trip_b_overlap;
	    strcpy( protocol_name, "blocking" );
	}
	else {  /* Assume -async */
	    f = round_trip_nb_overlap;
	    strcpy( protocol_name, "nonblocking" );
	}
	MsgSize = 0;
	SYArgGetInt( &argc, argv, 1, "-overlapmsgsize", &MsgSize );
	MsgCtx  = OverlapInit( proc1, proc2, MsgSize );
	/* Compute floating point lengths if requested */
	if (SYArgHasName( &argc, argv, 1, "-overlapauto")) {
	    OverlapSizes( MsgSize >= 0 ? MsgSize : 0, svals, MsgCtx );
	}
	strcat( protocol_name, "-overlap" );
	if (MsgSize >= 0) {
	    sprintf( cbuf, "-%d bytes", MsgSize );
	}
	else {
	    strcpy( cbuf, "-no msgs" );
	}
	strcat( protocol_name, cbuf );
	TimeScale = 0.5;
	RateScale = 2.0;
    }
    else if (SYArgHasName( &argc, argv, 1, "-memcpy" )) {
	f = memcpy_rate;
	MsgCtx     = 0;
	ChangeDist = 0;
	strcpy( protocol_name, "memcpy" );
	TimeScale = 1.0;
	RateScale = 1.0;
    }
    else {
	/* Pair by default */
	f = GetPairFunction( &argc, argv, protocol_name );
	MsgCtx = PairInit( proc1, proc2 );
	ChangeDist = PairChange;
	if (SYArgHasName( &argc, argv, 1, "-debug" ))
	    PrintPairInfo( MsgCtx );
	TimeScale = 0.5;
	RateScale = 2.0;
    }
    first = svals[0];
    last  = svals[1];
    incr  = svals[2];
    if (incr == 0) incr = 1;

/*
   Finally, we are ready to run the tests.  We want to report times as
   the times for a single link, and rates as the aggregate rate.
   To do this, we need to know how to scale both the times and the rates.

   Times: scaled by the number of one-way trips measured by the base testing
   code.  This is often 2 trips, or a scaling of 1/2.

   Rates: scaled by the number of simultaneous participants (as well as
   the scaling in times).  Compute the rates based on the updated time,
   then multiply by the number of participants.  Note that, for a single
   sender, time and rate are inversely proportional (that is, if TimeScale
   is 0.5, RateScale is 2.0).

 */

    start_time = MPI_Wtime();

/* If the distance flag is set, we look at a range of distances.  Otherwise,
   we just use the first and last processor */
    if (doinfo && __MYPROCID == 0) {
	HeaderGraph( outctx, protocol_name, (char *)0, units );
    }
    if(distance_flag) {
	for(distance=1;distance<GetMaxIndex();distance++) {
	    proc2 = GetNeighbor( 0, distance, 0 );
	    if (ChangeDist)
		(*ChangeDist)( distance, MsgCtx );
	    time_function(reps,first,last,incr,proc1,proc2,f,outctx,
			  autosize,autodx,autorel,MsgCtx);
	    ClearTimes();
	}
    }
    else{
	time_function(reps,first,last,incr,proc1,proc2,f,outctx,
		      autosize,autodx,autorel,MsgCtx);
    }

/*
   Generate the "end of page".  This allows multiple distance graphs on the
   same plot
 */
    if (doinfo && __MYPROCID == 0)
	EndPageGraph( outctx );

    MPI_Finalize();
    return 0;
}

/*
   This is the basic routine for timing an operation.

   Input Parameters:
.  reps - Basic number of times to run basic test (see below)
.  first,last,incr - length of data is first, first+incr, ... last
         (if last != first + k * incr, then actual last value is the
         value of first + k * incr that is <= last and such that
         first + (k+1) * incr > last, just as you'd expect)
.  proc1,proc2 - processors to participate in communication.  Note that
         all processors must call because we use global operations to
         manage some operations, and we want to avoid using process-subset
         operations (supported in Chameleon) to simplify porting this code
.  f  -  Routine to call to run a basic test.  This routine returns the time
         that the test took in seconds.
.  outctx -  Pointer to output context
.  autosize - If true, the actual sizes are picked automatically.  That is
         instead of using first, first + incr, ... , the routine choses values
         of len such that first <= len <= last and other properties, given
         by autodx and autorel, are satisfied.
.  autodx - Parameter for TST1dauto, used to set minimum distance between
         test sizes.  4 (for 4 bytes) is good for small values of last
.  autorel - Relative error tolerance used by TST1dauto in determining the
         message sizes used.
.  msgctx - Context to pass through to operation routine
 */
void time_function(reps,first,last,incr,proc1,proc2,f,outctx,
              autosize,autodx,autorel,msgctx)
int    reps,first,last,incr,proc1,proc2,autosize,autodx;
double autorel;
double (* f)();
void   *outctx;
void   *msgctx;
{
    int    len,distance,myproc;
    double mean_time;
    double s, r;
    double T1, T2;
    int    Len1, Len2;

    myproc   = __MYPROCID;
    distance = ((proc1)<(proc2)?(proc2)-(proc1):(proc1)-(proc2));

    /* Run test, using either the simple direct test or the automatic length
     test */
    ntest = 0;
    if (autosize) {
	int    maxvals = 256, nvals, i;
	int    dxmax;
	TwinTest ctx;
	TwinResults *results;

	/* We should really set maxvals as 2+(last-first)/autodx */
	results	 = (TwinResults *)malloc((unsigned)
					 (maxvals * sizeof(TwinResults) ));
	if (!results)exit(1);;
	ctx.reps	 = reps;
	ctx.f	 = f;
	ctx.msgctx = msgctx;
	ctx.proc1	 = proc1;
	ctx.proc2	 = proc2;
	ctx.t1	 = 0.0;
	ctx.t2	 = 0.0;
	ctx.len1	 = 0;
	ctx.len2	 = 0;

	/* We need to pick a better minimum resolution */
	dxmax = (last - first) / 16;
	/* make dxmax a multiple of 4 */
	dxmax = (dxmax & ~0x3);
	if (dxmax < 4) dxmax = 4;

	nvals = TSTAuto1d( (double)first, (double)last, (double)autodx,
			   (double)dxmax, autorel, 1.0e-10,
			   results, sizeof(TwinResults),
			   maxvals, GeneralF, &ctx );
	if (myproc == 0) {
	    TSTRSort( results, sizeof(TwinResults), nvals );
	    for (i = 0; i < nvals; i++) {
		DataoutGraph( outctx, proc1, proc2, distance,
			      (int)results[i].len, results[i].t * TimeScale,
			      results[i].mean_time * TimeScale,
			      results[i].rate * RateScale,
			      results[i].smean_time * TimeScale,
			      results[i].max_time * TimeScale );
	    }
	}
	free(results );
    }
    else {
	T1 = 0;
	T2 = 0;
	if (nsizes) {
	    int i;
	    for (i=0; i<nsizes; i++) {
		len = sizelist[i];
		RunATest( len, &Len1, &Len2, &T1, &T2, &reps, f,
			  myproc, proc1, proc2, distance, outctx, msgctx );
	    }
	}
	else {
	    for(len=first;len<=last;len+=incr){
		RunATest( len, &Len1, &Len2, &T1, &T2, &reps, f,
			  myproc, proc1, proc2, distance, outctx, msgctx );

	    }
	}
    }
/* Generate C.It output */
    if (doinfo && myproc == 0) {
	RateoutputGraph( outctx,
			 sumlen, sumtime, sumlentime, sumlen2, sumtime2,
			 ntest, &s, &r );
	DrawGraph( outctx, first, last, s, r );
    }

}



/*****************************************************************************
   Utility routines
 *****************************************************************************/

/* This runs a test for a given length */
void RunATest( len, Len1, Len2, T1, T2, reps, f,
	      myproc, proc1, proc2, distance, outctx, msgctx )
int len, *Len1, *Len2, *reps, myproc, proc1, proc2, distance;
double *T1, *T2;
double (*f)();
void   *outctx, *msgctx;
{
    double mean_time, t, rate;
    double tmax, tmean;

    if (AutoReps) {
	*reps = GetRepititions( *T1, *T2, *Len1, *Len2, len, *reps );
    }
    t = RunSingleTest( f, *reps, len, msgctx, &tmax, &tmean );
    mean_time = t;
    mean_time = mean_time / *reps;  /* take average over trials */
    if (mean_time > 0.0)
	rate      = ((double)len)/mean_time;
    else
	rate      = 0.0;
    if(myproc==0) {
	DataoutGraph( outctx, proc1, proc2, distance, len,
		      t * TimeScale, mean_time * TimeScale,
		      rate * RateScale, tmean * TimeScale, tmax * TimeScale );
    }

    *T1   = *T2;
    *Len1 = *Len2;
    *T2   = mean_time;
    *Len2 = len;
}

/*
   This routine computes a good number of repititions to use based on
   previous computations
 */
int ComputeGoodReps( t1, len1, t2, len2, len )
double t1, t2;
int    len1, len2, len;
{
    double s, r;
    int    reps;

    r = (t2 - t1) / (len2 - len1);
    s = t1 - r * len1;

    if (s <= 0.0) s = 0.0;
    reps = Tgoal / (s + r * len );

    if (reps < 1) reps = 1;

/*
printf( "Reps = %d (%d,%d,%d)\n", reps, len1, len2, len ); fflush( stdout );
 */
    return reps;
}


/*
  This runs the tests for a single size.  It adapts to the number of
  tests necessary to get a reliable value for the minimum time.
  It also keeps track of the average and maximum times (which are unused
  for now).

  We can estimate the variance of the trials by using the following
  formulas:

  variance = (1/N) sum (t(i) - (s+r n(i))**2
           = (1/N) sum (t(i)**2 - 2 t(i)(s + r n(i)) + (s+r n(i))**2)
	   = (1/N) (sum t(i)**2 - 2 s sum t(i) - 2 r sum t(i)n(i) +
	      sum (s**2 + 2 r s n(i) + r**2 n(i)**2))
  Since we compute the parameters s and r, we need only maintain
              sum t(i)**2
              sum t(i)n(i)
              sum n(i)**2
  We already keep all of these in computing the (s,r) parameters; this is
  simply a different computation.

  In the case n == constant (that is, inside a single test), we can use
  a similar test to estimate the variance of the individual measurements.
  In this case,

  variance = (1/N) sum (t(i) - s**2
           = (1/N) sum (t(i)**2 - 2 t(i)s + s**2)
	   = (1/N) (sum t(i)**2 - 2 s sum t(i) + sum s**2)
  Here, s = sum t(i)/N
  (For purists, the divison should be slightly different from (1/N) in the
  variance formula.  I'll deal with that later.)

  tmax = max time observed
  tmean = mean time observed
 */
double RunSingleTest( f, reps, len, msgctx, tmaxtime, tmean )
double (*f)();
int    reps;
void   *msgctx;
double *tmaxtime, *tmean;
{
    int    flag, k, iwork, natmin;
    double t, tmin, mean_time, tmax, tsum;
    int    rank;

    flag   = 0;
    tmin   = 1.0e+38;
    tmax   = tsum = 0.0;
    natmin = 0;

    for (k=0; k<minreps && flag == 0; k++) {
	t = (* f) (reps,len,msgctx);
	if (__MYPROCID == 0) {
	    tsum += t;
	    if (t > tmax) tmax = t;
	    if (t < tmin) {
		tmin   = t;
		natmin = 0;
	    }
	    else if (minThreshTest < k && tmin * (1.0 + repsThresh) > t) {
		/* This time is close to the minimum; use this to decide
		   that we've gotten close enough */
		natmin++;
		if (natmin >= NatThresh)
		    flag = 1;
	    }
	}
	MPI_Allreduce(&flag, &iwork, 1, MPI_INT,MPI_SUM,MPI_COMM_WORLD );
	flag = iwork;

	/* Check for max time exceeded */
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	if (rank == 0 && MPI_Wtime() - start_time > max_run_time) {
	    fprintf( stderr, "Exceeded %f seconds, aborting\n", max_run_time );
	    MPI_Abort( MPI_COMM_WORLD, 1 );
	}
    }

    mean_time  = tmin / reps;
    sumlen     += len;
    sumtime    += mean_time;
    sumlen2    += ((double)len)*((double)len);
    sumlentime += mean_time * len;
    sumtime2   += mean_time * mean_time;
    ntest      ++;

    if (tmaxtime) *tmaxtime = tmax / reps;
    if (tmean)    *tmean    = (tsum / reps ) / k;
    return tmin;
}

int PrintHelp( argv )
char **argv;
{
  if (__MYPROCID != 0) return 0;
  fprintf( stderr, "%s - test individual communication speeds\n", argv[0] );
  fprintf( stderr,
"Test a single communication link by various methods.  The tests are \n\
combinations of\n\
  Protocol: \n\
  -sync        Blocking sends/receives    (default)\n\
  -async       NonBlocking sends/receives\n\
  -force       Ready-receiver (with a null message)\n\
  -persistant  Persistant communication (only with MPI)\n\
  -vector      Data is separated by constant stride (only with MPI)\n\
\n\
  Message data:\n\
  -cachesize n Perform test so that cached data is NOT reused\n\
\n\
  -vstride n   For -vector, set the stride between elements\n\
  Message pattern:\n\
  -roundtrip   Roundtrip messages         (default)\n\
  -head        Head-to-head messages\n\
    \n" );
  fprintf( stderr,
"  Message test type:\n\
  (if not specified, only communication tests run)\n\
  -overlap     Overlap computation with communication (see -size)\n\
  -overlapmsgsize nn\n\
               Size of messages to overlap with is nn bytes.\n\
  -bisect      Bisection test (all processes participate\n\
  -bisectdist n Distance between processes\n\
    \n" );
  fprintf( stderr,
"  Message sizes:\n\
  -size start end stride                  (default 0 1024 32)\n\
               Messages of length (start + i*stride) for i=0,1,... until\n\
               the length is greater than end.\n\
  -sizelist n1,n2,...\n\
               Messages of length n1, n2, etc are used.  This overrides \n\
               -size\n\
  -auto        Compute message sizes automatically (to create a smooth\n\
               graph.  Use -size values for lower and upper range\n\
  -autodx n    Minimum number of bytes between samples when using -auto\n\
  -autorel d   Relative error tolerance when using -auto (0.02 by default)\n");

  fprintf( stderr, "\n\
  Number of tests\n\
  -reps n      Number of times message is sent (default 1000)\n\
  -autoreps    Compute the number of times a message is sent automatically\n\
  -tgoal  d    Time that each test should take, in seconds.  Use with \n\
               -autoreps\n\
  -rthresh d   Fractional threshold used to determine when minimum time\n\
               has been found.  The default is 0.05.\n\
  -sample_reps n   Number of times a full test is run inorder to find the\n\
               minimum average time.  The default is 30\n\
\n" );
fprintf( stderr, "  -gop [ options ]:\n" );
PrintGOPHelp();
PrintGraphHelp();
PrintPatternHelp();
return 0;
}

/*
   Re-initialize the variables used to estimate the time that it
   takes to send data
 */
ClearTimes()
{
sumtime	   = 0.0;
sumlentime = 0.0;
sumlen	   = 0.0;
sumlen2	   = 0.0;
sumtime2   = 0.0;
ntest	   = 0;
}

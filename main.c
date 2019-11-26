#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]);
void timestamp ( );

int main(int argc, char *argv[])
{
    int max_step;
    int step = 0;
    int id;
    int ierr;

    int p;
    int primes;
    int primes_part;
    double wtime;

	double total_time;
	clock_t start, end;

    if (argc > 1) {
        max_step = atoi(argv[1]);
    } else max_step = 100;
    
    /*
        Initialize MPI.
    */
    ierr = MPI_Init(&argc, &argv);
    
    /*
        Get the number of processes.
    */
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &p);
    
    /*
        Determine this processes's rank.
    */
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == 0)
    {   
        start = clock();
        timestamp ( );
        printf("The number of processes is %d\n", p);
        printf("Number of steps: %d\n", max_step);
    }

    while (step <= max_step)
    {
        if (id == 0)
        {
            wtime = MPI_Wtime();
        }
        // ierr = MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // primes_part = prime_number(n, id, p);

        // ierr = MPI_Reduce(&primes_part, &primes, 1, MPI_INT, MPI_SUM, 0,
        //                   MPI_COMM_WORLD);

        if (id == 0)
        {
            wtime = MPI_Wtime() - wtime;
            printf("  %8d  %8d  %14f\n", n, primes, wtime);
        }
        // n = n * n_factor;
        step++;
    }

    /*
        Terminate MPI.
    */
    ierr = MPI_Finalize();
    
    /*
        Terminate.
    */
    if (id == 0)
    {
        printf("\n");
        printf("Master process:\n");
        printf("  Normal end of execution.\n");
        printf("\n");
        timestamp();
        total_time = ((double) (end - start)) / CLOCKS_PER_SEC;
	    //calulate total time
	    printf("\nTime: %f", total_time);
    }

    return 0;
}

void timestamp ( void )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
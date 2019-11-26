#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MAX_SIZE 1000

int main(int argc, char *argv[]);
void timestamp ( );
void shuffle(int *array, size_t n);
    
int main(int argc, char *argv[])
{
    int max_step = 100;
    int batch_size = 64;
    int step = 0;
    float lr = 0.001;
    
    int n_sample;
    int data_dim;

    int machine_id;
    int n_machines;
    int ierr;
    double wtime;

    // Read Hyperparams
    // if (argc > 1) {
    //     max_step = atoi(argv[1]);
    // } else max_step = 100;
    
    // Read matrix data 

    double **X = malloc(MAX_SIZE * sizeof(double *));
    for (int i = 0; i < MAX_SIZE; ++i)
        X[i] = malloc(MAX_SIZE * sizeof(double));

    double *Y = malloc(MAX_SIZE * sizeof(double));

    FILE *file;
    file = fopen("matrix", "r");
    
    fscanf(file, "%d", &n_sample);
    fscanf(file, "%d", &data_dim);
    
    data_dim = data_dim -1;
    double *W = malloc(data_dim * sizeof(double));
    
    int *index = malloc(n_sample*sizeof(int));
    for (int i=0;i<n_sample;i++) 
        index[i] = i;

    for (int i = 0; i < n_sample; i++) {
        for (int j = 0; j < data_dim; j++)
            if (!fscanf(file, "%lf", &X[i][j]))
                break;
        if (!fscanf(file, "%lf", &Y[i]))
            break;
    }

    fclose(file);
    
    /*
        Initialize MPI.
    */
    ierr = MPI_Init(&argc, &argv);
    
    /*
        Get the number of processes.
    */
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &n_machines);
    
    /*
        Determine this processes's rank.
    */
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &machine_id);

    if (machine_id == 0)
    {   
        timestamp ( );
        printf("X data\n");
        for (int i=0;i<n_sample;i++) {
            for (int j=0;j<data_dim;j++) {
                printf("%lf ",X[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        printf("Y data\n");
        for (int i=0;i<n_sample;i++) {
            printf("%lf ",Y[i]);
        }
        printf("\n");
        
        printf("The number of processes is %d\n", n_machines);
        printf("Number of steps: %d\n", max_step);
        for (int i=0;i<data_dim;i++) {
            W[i] = (double)rand()/(double)(RAND_MAX);
        }
    }

    while (step <= max_step)
    {
        if (machine_id == 0)
        {
            wtime = MPI_Wtime();
            shuffle(index, n_sample);
            for(int i =0;i<n_sample;i++)
                printf("%d ",index[i]);
        }
        // suffle data


        // split data 
        // ierr = MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // cal grad
        // primes_part = prime_number(n, id, p);

        //combine grad
        // ierr = MPI_Reduce(&primes_part, &primes, 1, MPI_INT, MPI_SUM, 0,
        //                   MPI_COMM_WORLD);

        if (machine_id == 0)
        {
            wtime = MPI_Wtime() - wtime;
            printf ( "Step %d time %14f\n", step, wtime );
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
    if (machine_id == 0)
    {
        printf("W data\n");
        for (int i=0;i<data_dim;i++) {
            printf("%lf ",W[i]);
        }
        printf("\n");

        printf("\n");
        printf("Master process:\n");
        printf("  Normal end of execution.\n");
        printf("\n");
        timestamp();
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

void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}
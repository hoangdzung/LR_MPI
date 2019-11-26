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
    int max_step = 2;
    int batch_size = 64;
    int step = 0;
    float lr = 0.001;
    
    int n_samples;
    int data_dim;

    int machine_id;
    int n_machines;
    int ierr;
    double wtime;

    int dummpy = 0;
    // Read Hyperparams
    // if (argc > 1) {
    //     max_step = atoi(argv[1]);
    // } else max_step = 100;
    
    // Read matrix data 

    double **X = (double **) malloc(MAX_SIZE * sizeof(double *));
    for (int i = 0; i < MAX_SIZE; ++i)
        X[i] = malloc(MAX_SIZE * sizeof(double));

    double *Y = (double *) malloc(MAX_SIZE * sizeof(double));

    FILE *file;
    file = fopen("matrix", "r");
    
    fscanf(file, "%d", &n_samples);
    fscanf(file, "%d", &data_dim);
    
    data_dim = data_dim -1;
    double *W = (double *) malloc(data_dim * sizeof(double));
    double *grad = (double *) malloc(data_dim * sizeof(double));
    double *part_grad = (double *) malloc(data_dim * sizeof(double));

    int *index = (int *) malloc(n_samples*sizeof(int));

    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < data_dim; j++)
            if (!fscanf(file, "%lf", &X[i][j]))
                break;
        if (!fscanf(file, "%lf", &Y[i]))
            break;
    }

    fclose(file);
    
    int n_batches = (int) n_samples/batch_size;
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

    if (machine_id==0) {
        for (int i=0;i<n_samples;i++) 
            index[i] = i;
    }

    if (machine_id == 0)
    {   
        timestamp ( );
        printf("X data\n");
        for (int i=0;i<n_samples;i++) {
            for (int j=0;j<data_dim;j++) {
                printf("%lf ",X[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        printf("Y data\n");
        for (int i=0;i<n_samples;i++) {
            printf("%lf ",Y[i]);
        }
        printf("\n");
        
        printf("The number of processes is %d\n", n_machines);
        printf("Number of steps: %d\n", max_step);

        // Weight init
        for (int i=0;i<data_dim;i++) {
            W[i] = (double)rand()/(double)(RAND_MAX);
        }

        // Print init weight
        for(int i=0;i<data_dim;i++) 
            printf("Init W %lf\n", W[i]);
    }

    // BCast init weight to all machine
    ierr = MPI_Bcast (W, data_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  
    while (step < max_step)
    {
        if (machine_id == 0)
        {
            wtime = MPI_Wtime();
            shuffle(index, n_samples);
        }

        // BCast shuffled index to all machine
        ierr = MPI_Bcast (index, n_samples, MPI_INT, 0, MPI_COMM_WORLD );
        
        // split data 
        // ierr = MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // cal grad
        // primes_part = prime_number(n, id, p);


        for (int i =0;i<data_dim;i++) {
            part_grad[i] = step;
        }

        // //combine grad
        ierr = MPI_Reduce(&part_grad, &grad, data_dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (machine_id == 0)
        {
            for (int i =0;i<data_dim;i++) {
                W[i] -= lr * grad[i];
            }
            wtime = MPI_Wtime() - wtime;
            printf ( "Step %d time %14f\n", step, wtime );
        }
        ierr = MPI_Bcast (W, data_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD );
        step++;
    }
    for(int i=0;i<data_dim;i++) 
        printf("Machine %d: W %lf\n", machine_id, W[i]);
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
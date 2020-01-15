#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MAX_SIZE 1000

int main(int argc, char *argv[]);
void timestamp ( );
void shuffle(int *array, size_t n);
double sigmoid(double x);

int main(int argc, char *argv[])
{
    int debug = 1;
    int max_step = 10000;
    int batch_size = 3;
    int step = 0;
    double lr = 0.0001;
    
    int n_samples;
    int data_dim;

    int machine_id;
    int n_machines;
    int ierr;
    double wtime;

    int dummpy = 0;
    // Read Hyperparams
    if (argc > 1) {
        debug = atoi(argv[1]);
    }
    
    // Read matrix data 

    double **X = (double **) malloc(MAX_SIZE * sizeof(double *));
    for (int i = 0; i < MAX_SIZE; ++i)
        X[i] = malloc(MAX_SIZE * sizeof(double));

    double *Y = (double *) malloc(MAX_SIZE * sizeof(double));

    FILE *file;
    file = fopen("matrix2", "r");
    
    fscanf(file, "%d", &n_samples);
    fscanf(file, "%d", &data_dim);
    
    int n_batches = (int) n_samples/batch_size;
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

    int batch_size_per_machine = (int) batch_size/n_machines;

    double **X_batch = (double **) malloc(batch_size_per_machine * sizeof(double *));
    for (int i = 0; i < batch_size_per_machine; ++i)
        X_batch[i] = malloc(data_dim * sizeof(double));
        
    double *Y_batch = (double *) malloc(batch_size_per_machine * sizeof(double));
    double *temp_values = (double *) malloc(batch_size_per_machine * sizeof(double));

    if (machine_id == 0)
    {   
        timestamp ( );
        if (debug) {
            printf("\nX data\n");
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
            printf("\n\n");
            
            printf("Number of processes: %d\n", n_machines);
            printf("Number of steps: %d\n", max_step);
        }
        
        // Index init
        for (int i=0;i<n_samples;i++) 
            index[i] = i;

        // Weight init
        for (int i=0;i<data_dim;i++) {
            W[i] = (double)rand()/(double)(RAND_MAX);
        }

        if (debug) {
            printf("Number of batch: %d\n\n",n_batches);
            // Print init weight
            for(int i=0;i<data_dim;i++) 
                printf("Init W %lf\n", W[i]);
            printf("\n");
        }
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

        int batch_id = 0;
        int start = 0;
        while(batch_id < n_batches) {
            start = batch_id * batch_size;
            for (int i=0; i<batch_size_per_machine;i++) {
                for (int j=0;j<data_dim;j++)
                    X_batch[i][j] = X[start+machine_id*batch_size_per_machine+i][j];
                Y_batch[i] = Y[start+machine_id*batch_size_per_machine+i];
            }

            for(int i=0; i<batch_size_per_machine; ++i)
            {
                temp_values[i] = 0;
            }
            // XW-Y
            for(int i=0; i<batch_size_per_machine; ++i) {
                for(int j =0; j<data_dim; ++j)
                {
                    temp_values[i]+=X_batch[i][j]*W[j];
                }
                temp_values[i] = sigmoid(temp_values[i]);
                temp_values[i] -= Y_batch[i];
            }
            // X.T(XW-Y)
            for(int i=0; i<data_dim; ++i) {
                for(int j=0; j<batch_size_per_machine; ++j)
                {
                    part_grad[i]+=X_batch[j][i]*temp_values[j];
                }
            }

            // for (int i=0;i<batch_size_per_machine;i++) {
            //     printf("%d X[%d] %lf %lf\n",machine_id, i, X_batch[i][0],X_batch[i][1]);
            // }

            // for (int i=0;i<batch_size_per_machine;i++) {
            //     printf("%d Y[%d] %lf \n",machine_id, i, Y_batch[i]);
            // }


            // for (int i =0;i<data_dim;i++) {
            //     part_grad[i] = step;
            // }

            /*
                Combine grad and update weight using REDUCE
            */
            /* ===================================================================================*/
            // ierr = MPI_Reduce(part_grad, grad, data_dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            // if (machine_id == 0)
            // {
            //     for (int i =0;i<data_dim;i++) {
            //         W[i] = W[i] -lr * grad[i];
            //     }
            //     wtime = MPI_Wtime() - wtime;
            //     // printf ( "Step %d time %14f\n", step, wtime );
            // }
            // // BCast updated weight to all machine
            // ierr = MPI_Bcast (W, data_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD );
            /* ===================================================================================*/

            /*
                Combine grad and update weight using REDUCE
            */
            /* ===================================================================================*/
            ierr = MPI_Allreduce(part_grad, grad, data_dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            for (int i =0;i<data_dim;i++) {
                W[i] = W[i] -lr * grad[i];
            }
            /* ===================================================================================*/
            if (debug) {
                for(int i=0;i<data_dim;i++) 
                    printf("Step %d Machine %d: W %lf\n", step, machine_id, W[i]);  
            } 
            batch_id++;
        }
        step++;
    }
    if (debug) {
        for(int i=0;i<data_dim;i++) 
            printf("Machine %d: W %lf\n", machine_id, W[i]);
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

double sigmoid(double x)
{
     double exp_value;
     double return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}
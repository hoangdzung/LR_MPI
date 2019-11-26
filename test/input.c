#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main()
{
    int i, j, n, m;

    double **mat = malloc(100 * sizeof(double *));
    for (i = 0; i < 100; ++i)
        mat[i] = malloc(100 * sizeof(double));

    FILE *file;
    file = fopen("a", "r");
    fscanf(file, "%d", &n);
    fscanf(file, "%d", &m);

    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            if (!fscanf(file, "%lf", &mat[i][j]))
                break;
    for (i = 0; i < n; i++){
        for (j = 0; j < m; j++)
            printf("%lf ", mat[i][j]); 
        printf("\n");
    }
    fclose(file);
}
    #include <stdio.h>
    int main()
    {
        int X[3][4] = { {1, 1, 1, 1}, 
                    {2, 2, 2, 2}, 
                    {4, 4, 4, 4}}; 
  
        int W[4] = {1,2,3,4}; 
        int y[3] = {1,2,4}; 
    
   
        int temp[3];
        int result[4];
        int batch_size=3, dim=4, i, j;
       
        for(i=0; i<batch_size; ++i)
            {
                temp[i] = 0;
            }

        for(i=0; i<batch_size; ++i) {
            for(j=0; j<dim; ++j)
            {
                temp[i]+=X[i][j]*W[j];
            }
            temp[i] -= y[i];
        }

        for(i=0; i<dim; ++i) {
            for(j=0; j<batch_size; ++j)
            {
                result[i]+=X[j][i]*temp[j];
            }
        }

        // Displaying the result
        printf("\nOutput Matrix:\n");
        for(i=0; i<batch_size; ++i)
            {
                printf("%d  ", temp[i]);
            }
        printf("\n");
        for(i=0; i<dim; ++i)
            {
                printf("%d  ", result[i]);
            }
        return 0;
    }
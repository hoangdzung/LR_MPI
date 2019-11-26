#include <stdlib.h>
#include <stdio.h>
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

int main() {
    int a[5] = {1,2,3,4,5};
    shuffle(a, 4);
    for(int i=0;i<5;i++)
        printf("%d ",a[i]);
}
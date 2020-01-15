mpicc linear.c -o linear.o -lm
mpirun -np 2 linear.o

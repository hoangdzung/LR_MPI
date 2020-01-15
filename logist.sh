mpicc logist.c -o logist.o -lm
mpirun -np 2 logist.o

mpicc linear.c -o linear.o -lm
# mpirun -np 2 linear.o
# mpirun --hostfile ../hostfile2 -np 3 linear.o
mpirun -host localhost:1,192.168.1.30:1 ./linear.o
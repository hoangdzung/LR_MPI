# LR_MPI
Linear Regression and Logistic Regression implementation using C/OpenMPI

## How to run
The input files are hardcode in `logist.c` and `linear.c`, which are `linear.train`, `linear.test`, `logits.train`, `logits.test`
### Input file format 
```alias
Number-of-examples(N) Data-dim(D)
X1_1 X1_2 ... X1_(D-1) Y1
...
XN_1 XN_2 ... X_(D-1) YN
```
### Linear Regression
Compile and run:
``` bash
./linear.sh
```
### Logistic Regression

Compile and run:
``` bash
./logist.sh
```

## Reference
Algorithm in matrix form:
1. [Linear Regression](https://machinelearningcoban.com/2016/12/28/linearregression/#-nghiem-cho-bai-toan-linear-regression)
2. [Logistic Regression](https://machinelearningcoban.com/2017/01/27/logisticregression/#cong-thuc-cap-nhat-cho-logistic-sigmoid-regression)
 

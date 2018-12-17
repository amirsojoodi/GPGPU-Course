#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<sys/time.h>
#include<cuda.h>

#define OPERATION(X) (X)*13+(X)/13+(X)*(X)+(X)*((X)-13)

#define RANDOM_NUMBER_MAX 1000

//Macro for checking cuda errors following a cuda launch or api call
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);												\
	} }


void set_clock();

double get_elapsed_time();

void validate(double *a, double *b, int length);

void initialize_data_random(double **data, int data_size);

void initialize_data_zero(double **data, int data_size);

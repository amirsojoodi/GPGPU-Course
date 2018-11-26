#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<sys/time.h>
#include<omp.h>

void set_clock();

double get_elapsed_time();

void validate(double *a, double *b, int length);

void initialize_data_random(double **data, int data_size);

void initialize_data_zero(double **data, int data_size);

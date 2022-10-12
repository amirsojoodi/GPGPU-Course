#ifndef _PI_H
#define _PI_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

void set_clock();

double get_elapsed_time();

void validate(double *a, double *b, int length);

void initialize_data_random(double **data, int data_size);

void initialize_data_zero(double **data, int data_size);

#endif

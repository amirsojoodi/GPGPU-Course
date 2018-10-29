#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<pthread.h>
#include<time.h>
#include<sys/time.h>

#define MIN(a,b) ((a) < (b) ? a : b)

#define RANGE 1000

extern pthread_mutex_t *locks;

typedef struct argument{
	int *a;
	int *hist;
	int *local_hist;
	int tid;
	int size;
	int hist_size;
	int num_thread;
} argument;

int *sequential_naive_histogram(int*, int*, int, int);

void first_parallel_histogram(int*, int*, pthread_t*, argument*, int, int, int);

void second_parallel_histogram(int*, int*, pthread_t*, argument*, int, int, int);

void third_parallel_histogram(int*, int*, pthread_t*, argument*, int, int, int);

void validate(int *a, int *b, int length);

void initialize_data_random(int **data, int data_size);

void initialize_data_zero(int **data, int data_size);

void initialize_threads(pthread_t **threads, int num_thread);

void initialize_arguments(argument ** arguments, int num_thread);

void initialize_locks(pthread_mutex_t **locks, int num_lock);

void destroy_locks(pthread_mutex_t *locks, int num_lock);


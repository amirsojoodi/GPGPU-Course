#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H
#include"pi.h"
#endif

struct timeval start, end;

void set_clock(){
	gettimeofday(&start, NULL);
}

double get_elapsed_time(){
	gettimeofday(&end, NULL);
	double elapsed = (end.tv_sec - start.tv_sec) * 1000000.0;
	elapsed += end.tv_usec - start.tv_usec;
	return elapsed;
}

void validate(double *a, double *b, int length) {
	for (int i = 0; i < length; ++i) {
		if (a[i] != b[i]) {
			printf("Different value detected at position: %d,"
					" expected %.15f but get %.15f\n", i, a[i], b[i]);
			return;
		}
	}
	printf("Tests PASSED successfully! There is no differences \\:D/\n");
}

void initialize_data_random(double **data, int data_size){

	static time_t t;
	srand((unsigned) time(&t));

	*data = (double *)malloc(sizeof(double) * data_size);    
	for(int i = 0; i < data_size; i++){
		(*data)[i] = rand();
	}   
}   

void initialize_data_zero(double **data, int data_size){
	*data = (double *)malloc(sizeof(double) * data_size);
	memset(*data, 0, data_size*sizeof(double));
}   


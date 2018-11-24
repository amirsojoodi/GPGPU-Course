#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H
#include"pi.h"
#endif

int main(int argc, char *argv[]){

	double elapsed_time;
	double *data;
	double serialPI, parallelPI, step;
	
	if(argc != 3){
		printf("Correct way to execute this program is:\n");
		printf("./pi stepSize numberOfThreads\n");
		return 1;
	}

	int data_size = atoi(argv[1]);
	int num_thread = atoi(argv[2]);
	
	serialPI = 0;
	parallelPI = 0;

	// Sequential histogram
	set_clock();

	// COMPLETE HERE

    elapsed_time = get_elapsed_time();

	printf("Naive PI calculation time: %.4fms\n", elapsed_time / 1000);

	// Openmp Parallel histogram
	set_clock();
	
	// COMPLETE HERE

    elapsed_time = get_elapsed_time();

    printf("-> Openmp PI calculation time: %.4fms\n", elapsed_time / 1000);

    #ifdef  TEST
    validate(&serialPI, &parallelPI, 1);
    #endif

	free(data);

	return 0;
}

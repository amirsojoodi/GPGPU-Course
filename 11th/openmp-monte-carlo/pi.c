#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H
#include"pi.h"
#endif

int main(int argc, char *argv[]){

	double elapsed_time;
	double serialPI = 0, parallelPI = 0;
	
	if(argc != 3){
		printf("Correct way to execute this program is:\n");
		printf("./pi stepNum numberOfThreads\n");
		return 1;
	}

	int num_steps = atoi(argv[1]);
	int num_thread = atoi(argv[2]);
	
	// Sequential histogram
	set_clock();

	// COMPLETE HERE
	time_t t;
	srand((unsigned) time(&t));

	double x = 0, y = 0;
	double sum = 0;
	int count_inside = 0;
	for(int i = 0; i < num_steps; i++){
		x = (double)rand()/RAND_MAX;
		y = (double)rand()/RAND_MAX;
		sum = x * x + y * y;
		if (sum <= 1){
			count_inside++;
		}
	}
	serialPI = (count_inside * 4.0) / num_steps;

    elapsed_time = get_elapsed_time();

	printf("Naive PI calculation time: %.4fms\n", elapsed_time / 1000);

	// Openmp Parallel histogram
	set_clock();
	
	// COMPLETE HERE
	omp_set_num_threads(num_thread);

	// Sequential histogram
	set_clock();
	int count_inside_shared = 0;
	#pragma omp parallel 
	{
		double x = 0, y = 0;
		double sum = 0;
		int count_inside = 0;
		int seed = 23 * omp_get_thread_num();
		#pragma omp for
		for(int i = 0; i < num_steps; i++){
			x = (double)rand_r(&seed)/RAND_MAX;
			y = (double)rand_r(&seed)/RAND_MAX;
			sum = x * x + y * y;
			if (sum <= 1){
				count_inside++;
			}
		}

		#pragma omp atomic
		count_inside_shared += count_inside;
	}

	parallelPI = (count_inside_shared * 4.0) / num_steps;
	
    elapsed_time = get_elapsed_time();

    printf("-> Openmp PI calculation time: %.4fms\n", elapsed_time / 1000);

    #ifdef  TEST
    validate(&serialPI, &parallelPI, 1);
    #endif

	return 0;
}

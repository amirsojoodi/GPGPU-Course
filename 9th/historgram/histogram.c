#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H
#include"histogram.h"
#endif

pthread_mutex_t *locks;

int main(int argc, char *argv[]){

	struct timeval start, end;
	int *data, *serial_hist, *parallel_hist;
	pthread_t *threads;
	argument *arguments;
	
	if(argc != 4){
		printf("Correct way to execute this program is:\n");
		printf("./histogram arraySize histogramSize numberOfThreads\n");
		return 1;
	}

	int data_size = atoi(argv[1]);
	int hist_size = atoi(argv[2]);
	int num_thread = atoi(argv[3]);
	
	initialize_data_random(&data, data_size);
	initialize_data_zero(&serial_hist, hist_size);

	initialize_data_zero(&parallel_hist, hist_size);
	initialize_threads(&threads, num_thread);
	initialize_arguments(&arguments, num_thread);
	initialize_locks(&locks, hist_size);

	// Sequential histogram
	gettimeofday(&start, NULL);

    sequential_naive_histogram(data, serial_hist, data_size, hist_size);

    gettimeofday(&end, NULL);

    double diff = (end.tv_sec - start.tv_sec) * 1000000.0 +
        (end.tv_usec - start.tv_usec);

    printf("Naive Histogram time calculation duration: %.4fms\n", diff / 1000);

	// First Parallel histogram
    gettimeofday(&start, NULL);
    
	first_parallel_histogram(data, parallel_hist, threads, arguments, data_size, hist_size, num_thread);

	gettimeofday(&end, NULL);

    diff = (end.tv_sec - start.tv_sec) * 1000000.0
            + (end.tv_usec - start.tv_usec);
    printf("-> 1st Parallel Histogram time calculation duration: %.4fms\n", diff / 1000);

    #ifdef  TEST
    validate(serial_hist, parallel_hist, hist_size);
    #endif
	
	// Second Parallel histogram
	memset(parallel_hist, 0, hist_size * sizeof(int));

    gettimeofday(&start, NULL);
    
	second_parallel_histogram(data, parallel_hist, threads, arguments, data_size, hist_size, num_thread);

	gettimeofday(&end, NULL);

    diff = (end.tv_sec - start.tv_sec) * 1000000.0
            + (end.tv_usec - start.tv_usec);
    printf("-> 2nd Parallel Histogram time calculation duration: %.4fms\n", diff / 1000);

    #ifdef  TEST
    validate(serial_hist, parallel_hist, hist_size);
    #endif

	// Third Parallel histogram
	memset(parallel_hist, 0, hist_size * sizeof(int));

    gettimeofday(&start, NULL);
    
	third_parallel_histogram(data, parallel_hist, threads, arguments, data_size, hist_size, num_thread);

	gettimeofday(&end, NULL);

    diff = (end.tv_sec - start.tv_sec) * 1000000.0
            + (end.tv_usec - start.tv_usec);
    printf("-> 3rd Parallel Histogram time calculation duration: %.4fms\n", diff / 1000);

    #ifdef  TEST
    validate(serial_hist, parallel_hist, hist_size);
    #endif

	destroy_locks(locks, hist_size);

	free(data);
	free(serial_hist);
	free(parallel_hist);
	free(threads);
	free(arguments);

	return 0;
}

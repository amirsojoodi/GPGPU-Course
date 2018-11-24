#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H
#include"histogram.h"
#endif

void openmp_histogram(int *data, int *parallel_hist, int data_size, int hist_size, int num_thread){
	

	int	bucketSize = RANGE / hist_size;
	
	omp_set_num_threads(num_thread);

	#pragma omp parallel shared(parallel_hist, bucketSize, data_size, hist_size)
	{
		int local_hist[hist_size];
		for(int i = 0; i < hist_size; i++){
			local_hist[i] = 0;
		}

		#pragma omp for		   
		for (int i = 0; i < data_size; i++) {
			local_hist[data[i] / bucketSize]++;
		}

		for (int i = 0; i < hist_size; i++) {
			#pragma omp atomic
			parallel_hist[i] += local_hist[i];
		}
	}
}

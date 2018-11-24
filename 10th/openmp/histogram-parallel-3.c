#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H
#include"histogram.h"
#endif

void *parallel_histogram_third(void *arg){
    argument targ = *(argument *)arg;

    int *a = targ.a;

    int chunk_size = targ.size / targ.num_thread;

    int si = chunk_size * targ.tid;
    int ei = si + chunk_size;

    if(targ.tid == targ.num_thread - 1){
        ei = targ.size;
    }

	int *hist_local = targ.local_hist;

    int bucketSize = RANGE / targ.hist_size;
    int index;
    for(int i = si; i < ei; i++){
        index = a[i] / bucketSize;
        hist_local[index]++;
    }
    pthread_exit(NULL);
}

void third_parallel_histogram(int *data, int *parallel_hist, pthread_t *threads, argument *arguments, int data_size, int hist_size, int num_thread){
	
	for(int i = 0; i < num_thread; i++){
        arguments[i].a = data;
        arguments[i].hist = parallel_hist;
        arguments[i].tid = i;
        arguments[i].size = data_size;
        arguments[i].hist_size = hist_size;
        arguments[i].num_thread = num_thread;

		initialize_data_zero(&(arguments[i].local_hist), hist_size);

        pthread_create(&threads[i], 0, parallel_histogram_third, (void *)&arguments[i]);
    }

	int status;
    for (int i = 0; i < num_thread; ++i) {
        pthread_join(threads[i], (void *)&status);
        for(int j = 0; j < hist_size; j++){
            parallel_hist[j] += arguments[i].local_hist[j];
        }
		free(arguments[i].local_hist);
    }
}

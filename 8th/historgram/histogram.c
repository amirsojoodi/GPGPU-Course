#include"histogram.h"

#define RANGE 1000

pthread_mutex_t lock;

void *parallel_histogram(void *arg){
	argument targ = *(argument *)arg;

	int *a = targ.a, *hist = targ.hist;
	
	int chunk_size = targ.size / targ.num_thread;

	int si = chunk_size * targ.tid; 
	int ei = si + chunk_size;

	if(targ.tid == targ.num_thread - 1){
		ei = targ.size;
	}	

	int bucketSize = RANGE / targ.hist_size;
    int index;
	for(int i = si; i < ei; i++){
		index = a[i] / bucketSize;
		pthread_mutex_lock(&lock);
        hist[index]++;
		pthread_mutex_unlock(&lock);
	}
	pthread_exit(NULL);
}

int *sequential_naive_histogram(int *a, int *serial_hist, int size, int hist_size){

	int bucketSize = RANGE / hist_size;

    for (int i = 0; i < size; i++) {
        serial_hist[a[i] / bucketSize]++;
    }
}

int main(int argc, char *argv[]){

	struct timeval start, end;
	int *a, *serial_hist, *parallel_hist;
	time_t t;
	pthread_t *threads;
	argument *arguments;

	if(argc != 4){
		printf("Correct way to execute this program is:\n");
		printf("./histogram arraySize histogramSize NumberOfThreads.\n");
		return 1;
	}

	int data_size = atoi(argv[1]);
	int hist_size = atoi(argv[2]);
	int num_thread = atoi(argv[3]);
	
	a = (int *)malloc(sizeof(int) * data_size);
	serial_hist = (int *)malloc(sizeof(int) * hist_size);
	parallel_hist = (int *)malloc(sizeof(int) * hist_size);
	
	threads = (pthread_t *)malloc(sizeof(pthread_t) * num_thread);
	arguments = (argument *)malloc(sizeof(argument) * num_thread);

	pthread_mutex_init(&lock, NULL);

	srand((unsigned) time(&t));

	for(int i = 0; i < data_size; i++){
		a[i] = rand() % RANGE;
	}

	gettimeofday(&start, NULL);

    sequential_naive_histogram(a, serial_hist, data_size, hist_size);

    gettimeofday(&end, NULL);

    double diff = (end.tv_sec - start.tv_sec) * 1000000.0 +
        (end.tv_usec - start.tv_usec);

    printf("Naive Histogram time calculation duration: %.4fms\n", diff / 1000);

    gettimeofday(&start, NULL);

	for(int i = 0; i < num_thread; i++){
		arguments[i].a = a;
		arguments[i].hist = parallel_hist;
		arguments[i].tid = i;
		arguments[i].size = data_size;
		arguments[i].hist_size = hist_size;
		arguments[i].num_thread = num_thread;
	
		pthread_create(&threads[i], 0, parallel_histogram, (void *)&arguments[i]);
	}
	
	int status;
    for (int i = 0; i < num_thread; ++i) {
        pthread_join(threads[i], (void *)&status);
    }

	gettimeofday(&end, NULL);

    diff = (end.tv_sec - start.tv_sec) * 1000000.0
            + (end.tv_usec - start.tv_usec);
    printf("Parallel Histogram time calculation duration: %.4fms\n", diff / 1000);

	#ifdef	TEST
	validate(serial_hist, parallel_hist, hist_size);
	#endif

	pthread_mutex_destroy(&lock);

	free(a);
	free(serial_hist);
	free(parallel_hist);
	free(threads);
	free(arguments);

	return 0;
}

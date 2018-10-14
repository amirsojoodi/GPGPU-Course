#include"vectorReduce.h"

void *parallel_vector_reduce(void *arg){
	argument targ = *(argument *)arg;

	int *a = targ.a;

	int si = (targ.size / targ.num_thread) * targ.tid; 
	int ei = MIN(si + targ.size / targ.num_thread, targ.size);

	int sum = 0;
	for(int i = si; i < ei; i++){
		sum += a[i];
	}
	targ.c[targ.tid] = sum;

	pthread_exit(NULL);
}

int sequential_naive_vectorReduce(int *a, int size){
	int sum = 0;
    for(int i = 0; i < size; i++){
		sum += a[i];
    }
	return sum;
}


int main(int argc, char *argv[]){

	struct timeval start, end;
	int *a, *c, *d;
	pthread_t *threads;
	argument *arguments;
	int result_serial, result_parallel;
	time_t t;

	if(argc != 3){
		printf("Correct way to execute this program is:\n");
		printf("./vectorReduce arraySize NumberOfThreads.\n");
		return 1;
	}

	int size = atoi(argv[1]);
	int num_thread = atoi(argv[2]);
	
	a = (int *)malloc(sizeof(int) * size);
	c = (int *)malloc(sizeof(int) * num_thread);
	threads = (pthread_t *)malloc(sizeof(pthread_t) * num_thread);
	arguments = (argument *)malloc(sizeof(argument) * num_thread);

	srand((unsigned) time(&t));

	result_serial = 0;
	result_parallel = 0;
	for(int i = 0; i <size; i++){
		a[i] = rand() % 10;
	}

	gettimeofday(&start, NULL);

    result_serial = sequential_naive_vectorReduce(a, size);

    gettimeofday(&end, NULL);

    double diff = (end.tv_sec - start.tv_sec) * 1000000.0 +
        (end.tv_usec - start.tv_usec);

    printf("Naive VectorReduce time calculation duration: %.4fms\n", diff / 1000);

    gettimeofday(&start, NULL);

	for(int i = 0; i < num_thread; i++){
		arguments[i].a = a;
		arguments[i].c = c;
		arguments[i].tid = i;
		arguments[i].size = size;
		arguments[i].num_thread = num_thread;
	
		pthread_create(&threads[i], 0, parallel_vector_reduce, (void *)&arguments[i]);
	}
	
	int status;
    for (int i = 0; i < num_thread; ++i) {
        pthread_join(threads[i], (void *)&status);
		result_parallel += c[i];
    }

	gettimeofday(&end, NULL);

    diff = (end.tv_sec - start.tv_sec) * 1000000.0
            + (end.tv_usec - start.tv_usec);
    printf("Parallel VectorReduce time calculation duration: %.4fms\n", diff / 1000);

	#ifdef	TEST
	validate(&result_serial, &result_parallel, 1);
	printf("Reduction result = %d\n", result_serial);
	#endif

	free(a);
	free(c);
	free(threads);
	free(arguments);

	return 0;
}

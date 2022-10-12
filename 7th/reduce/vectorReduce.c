#include "vectorReduce.h"

int result_parallel;
// pthread_mutex_t lock;

void *parallel_vector_reduce(void *targ) {
  // You should complete here.
  argument *arg = (argument *)targ;
  int *a = arg->a;
  int size = arg->size;
  int num_thread = arg->num_thread;
  int tid = arg->id;

  int chunk_size = size / num_thread;
  int si = chunk_size * tid;
  int ei = si + chunk_size;
  if (tid == num_thread - 1) {
    ei = size;
  }

  int my_sum = 0;
  for (int i = si; i < ei; i++) {
    my_sum += a[i];
  }

  arg->my_result = my_sum;
  //	pthread_mutex_lock(&lock);
  //	result_parallel += my_sum;
  //	pthread_mutex_unlock(&lock);

  pthread_exit(NULL);
}

int sequential_naive_vectorReduce(int *a, int size) {
  // You should complete here.
  int sum = 0;
  for (int i = 0; i < size; i++) {
    sum += a[i];
  }
  return sum;
}

int main(int argc, char *argv[]) {

  struct timeval start, end;
  int *a;
  pthread_t *threads;
  argument *arguments;
  int result_serial;
  time_t t;

  if (argc != 3) {
    printf("Correct way to execute this program is:\n");
    printf("./vectorReduce arraySize NumberOfThreads.\n");
    return 1;
  }

  int size = atoi(argv[1]);
  int num_thread = atoi(argv[2]);

  a = (int *)malloc(sizeof(int) * size);
  threads = (pthread_t *)malloc(sizeof(pthread_t) * num_thread);
  arguments = (argument *)malloc(sizeof(argument) * num_thread);

  //	pthread_mutex_init(&lock, NULL);

  srand((unsigned)time(&t));

  result_serial = 0;
  result_parallel = 0;
  for (int i = 0; i < size; i++) {
    a[i] = rand() % 10;
  }

  gettimeofday(&start, NULL);

  result_serial = sequential_naive_vectorReduce(a, size);

  gettimeofday(&end, NULL);

  double diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Naive VectorReduce time calculation duration: %.4fms\n", diff / 1000);

  gettimeofday(&start, NULL);

  for (int i = 0; i < num_thread; i++) {
    arguments[i].a = a;
    arguments[i].size = size;
    arguments[i].id = i;
    arguments[i].num_thread = num_thread;
    // You should complete here.

    pthread_create(&threads[i], 0, parallel_vector_reduce, &arguments[i]);
  }

  int status;
  for (int i = 0; i < num_thread; ++i) {
    pthread_join(threads[i], (void *)&status);
    result_parallel += arguments[i].my_result;
  }

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
  printf("Parallel VectorReduce time calculation duration: %.4fms\n",
         diff / 1000);

#ifdef TEST
  validate(&result_serial, &result_parallel, 1);
  printf("Reduction result = %d\n", result_serial);
#endif

  // pthread_mutex_destroy(&lock);
  free(a);
  // free(c);
  free(threads);
  free(arguments);

  return 0;
}

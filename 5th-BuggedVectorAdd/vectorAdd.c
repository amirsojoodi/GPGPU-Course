#include "vectorAdd.h"

void *parallel_vector_add(void *arg) {
  argument targ = *(argument *)arg;

  int *a = targ.a, *b = targ.b, *c = targ.c;

  int si = (targ.size / targ.num_threads) * targ.tid;
  int ei = min(si + targ.size / targ.num_threads, targ.size);

  for (int i = si; i < ei; i++) {
    c[i] = a[i] + b[i];
  }
  pthread_exit(NULL);
}

int *sequential_naive_vectorAdd(int *a, int *b, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = a[i] + b[i];
  }
  return a;
}

// vectorAddParallel Input_size thread
int main(int argc, char *argv[]) {

  struct timeval start, end;
  int *a, *b, *c, *d;
  time_t t;

  if (argc != 3) {
    printf("Correct way to execute this program is:\n");
    printf("./vectorAdd arraySize NumberOfThreads.\n");
    return 1;
  }

  int size = atoi(argv[1]);
  int num_threads = atoi(argv[2]);

  pthread_t threads[num_threads];
  argument arguments[num_threads];

  a = (int *)malloc(sizeof(int) * size);
  b = (int *)malloc(sizeof(int) * size);
  c = (int *)malloc(sizeof(int) * size);
  d = (int *)malloc(sizeof(int) * size);

  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  gettimeofday(&start, NULL);

  c = sequential_naive_vectorAdd(a, b, size);

  gettimeofday(&end, NULL);

  double diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Naive VectorAdd time calculation duration: %.4fms\n", diff / 1000);

  gettimeofday(&start, NULL);

  for (int i = 0; i < num_threads; i++) {
    arguments[i].a = a;
    arguments[i].b = b;
    arguments[i].c = c;
    arguments[i].tid = i;
    arguments[i].size = size;
    arguments[i].num_threads = num_threads;

    pthread_create(&threads[i], 0, parallel_vector_add, (void *)&arguments[0]);
  }

  int status;
  for (int i = 0; i < num_threads; ++i) {
    pthread_join(threads[i], (void *)&status);
  }

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
  printf("Parallel VectorAdd time calculation duration: %.4fms\n", diff / 1000);

#ifdef TEST
  validate(c, d, size);
#endif

  free(a);
  free(b);
  free(d);

  return 0;
}

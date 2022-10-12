#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define min(a, b) ((a) < (b) ? a : b)

typedef struct argument {
  int *a;
  int *b;
  int *c;
  int tid;
  int size;
  int num_thread;
} argument;

void validate(int *a, int *b, int length) {
  for (int i = 0; i < length; ++i) {
    if (a[i] != b[i]) {
      printf("Different value detected at position: %d,"
             "expected %d but get %d\n",
             i, a[i], b[i]);
      break;
    }
  }
}

void *naive_parallel_vector_add(void *arg) {
  argument targ = *(argument *)arg;

  int *a = targ.a, *b = targ.b, *c = targ.c;

  int si = (targ.size / targ.num_thread) * targ.tid;
  int ei = min(si + targ.size / targ.num_thread, targ.size);

  for (int i = si; i < ei; i++) {
    c[i] = a[i] * b[i];
    c[i] += (int)cos(a[i] + b[i]);
  }
  pthread_exit(NULL);
}

void sequential_naive_vectorAdd(int *a, int *b, int *c, int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
    c[i] += (int)cos(a[i] + b[i]);
  }
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
  int num_thread = atoi(argv[2]);

  pthread_t threads[num_thread];
  argument arguments[num_thread];

  a = (int *)malloc(sizeof(int) * size);
  b = (int *)malloc(sizeof(int) * size);
  c = (int *)malloc(sizeof(int) * size);
  d = (int *)malloc(sizeof(int) * size);

  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
    c[i] = 0;
    d[i] = 0;
  }

  gettimeofday(&start, NULL);

  sequential_naive_vectorAdd(a, b, c, size);

  gettimeofday(&end, NULL);

  double diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Naive VectorAdd time calculation duration: %.4fms\n", diff / 1000);

  gettimeofday(&start, NULL);

  for (int i = 0; i < num_thread; i++) {
    arguments[i].a = a;
    arguments[i].b = b;
    arguments[i].c = d;
    arguments[i].tid = i;
    arguments[i].size = size;
    arguments[i].num_thread = num_thread;

    pthread_create(&threads[i], 0, naive_parallel_vector_add,
                   (void *)&arguments[i]);
  }

  int status;
  for (int i = 0; i < num_thread; ++i) {
    pthread_join(threads[i], (void *)&status);
  }

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
  printf("Parallel VectorAdd time calculation duration: %.4fms\n", diff / 1000);

  validate(c, d, size);

  free(a);
  free(b);
  free(c);
  free(d);

  return 0;
}

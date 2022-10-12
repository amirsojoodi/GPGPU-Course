#include "histogram.h"

struct timeval start, end;

void set_clock() { gettimeofday(&start, NULL); }

double get_elapsed_time() {
  gettimeofday(&end, NULL);
  double elapsed = (end.tv_sec - start.tv_sec) * 1000000.0;
  elapsed += end.tv_usec - start.tv_usec;
  return elapsed;
}

void validate(int *a, int *b, int length) {
  for (int i = 0; i < length; ++i) {
    if (a[i] != b[i]) {
      printf("Different value detected at position: %d,"
             " expected %d but get %d\n",
             i, a[i], b[i]);
      return;
    }
  }
  printf("Tests PASSED successfully! There is no differences \\:D/\n");
}

void initialize_data_random(int **data, int data_size) {

  static time_t t;
  srand((unsigned)time(&t));

  *data = (int *)malloc(sizeof(int) * data_size);
  for (int i = 0; i < data_size; i++) {
    (*data)[i] = rand() % RANGE;
  }
}

void initialize_data_zero(int **data, int data_size) {
  *data = (int *)malloc(sizeof(int) * data_size);
  memset(*data, 0, data_size * sizeof(int));
}

void initialize_threads(pthread_t **threads, int num_thread) {
  *threads = (pthread_t *)malloc(sizeof(pthread_t) * num_thread);
}

void initialize_arguments(argument **arguments, int num_thread) {
  *arguments = (argument *)malloc(sizeof(argument) * num_thread);
}

void initialize_locks(pthread_mutex_t **locks, int num_lock) {

  *locks = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * num_lock);
  pthread_mutex_t *tmp_locks = *locks;

  for (int i = 0; i < num_lock; i++) {
    pthread_mutex_init(&tmp_locks[i], NULL);
  }
}

void destroy_locks(pthread_mutex_t *locks, int num_lock) {

  for (int i = 0; i < num_lock; i++) {
    pthread_mutex_destroy(&locks[i]);
  }
}

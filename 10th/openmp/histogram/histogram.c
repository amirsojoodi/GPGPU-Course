#include "histogram.h"

pthread_mutex_t *locks;

int main(int argc, char *argv[]) {

  double elapsed_time;
  int *data, *serial_hist, *parallel_hist;
  pthread_t *threads;
  argument *arguments;

  if (argc != 4) {
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
  set_clock();

  sequential_naive_histogram(data, serial_hist, data_size, hist_size);

  elapsed_time = get_elapsed_time();

  printf("Naive Histogram time calculation duration: %.4fms\n",
         elapsed_time / 1000);

  // Third Parallel histogram
  memset(parallel_hist, 0, hist_size * sizeof(int));

  set_clock();

  third_parallel_histogram(data, parallel_hist, threads, arguments, data_size,
                           hist_size, num_thread);

  elapsed_time = get_elapsed_time();

  printf("-> Pthread 3rd version Parallel Histogram time calculation duration: "
         "%.4fms\n",
         elapsed_time / 1000);

#ifdef TEST
  validate(serial_hist, parallel_hist, hist_size);
#endif

  // Openmp parallel histogram
  memset(parallel_hist, 0, hist_size * sizeof(int));

  set_clock();

  openmp_histogram(data, parallel_hist, data_size, hist_size, num_thread);

  elapsed_time = get_elapsed_time();

  printf("-> Openmp Parallel Histogram time calculation duration: %.4fms\n",
         elapsed_time / 1000);

#ifdef TEST
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

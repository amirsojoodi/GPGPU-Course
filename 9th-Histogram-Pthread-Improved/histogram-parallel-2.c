#include "histogram.h"

void *parallel_histogram_second(void *arg) {
  argument targ = *(argument *)arg;

  int *a = targ.a, *hist = targ.hist;

  int chunk_size = targ.size / targ.num_thread;

  int si = chunk_size * targ.tid;
  int ei = si + chunk_size;

  if (targ.tid == targ.num_thread - 1) {
    ei = targ.size;
  }

  int bucketSize = RANGE / targ.hist_size;
  int index;
  for (int i = si; i < ei; i++) {
    index = a[i] / bucketSize;
    pthread_mutex_lock(&locks[index]);
    hist[index]++;
    pthread_mutex_unlock(&locks[index]);
  }
  pthread_exit(NULL);
}

void second_parallel_histogram(int *data, int *parallel_hist,
                               pthread_t *threads, argument *arguments,
                               int data_size, int hist_size, int num_thread) {

  for (int i = 0; i < num_thread; i++) {
    arguments[i].a = data;
    arguments[i].hist = parallel_hist;
    arguments[i].tid = i;
    arguments[i].size = data_size;
    arguments[i].hist_size = hist_size;
    arguments[i].num_thread = num_thread;

    pthread_create(&threads[i], 0, parallel_histogram_second,
                   (void *)&arguments[i]);
  }

  int status;
  for (int i = 0; i < num_thread; ++i) {
    pthread_join(threads[i], (void *)&status);
  }
}

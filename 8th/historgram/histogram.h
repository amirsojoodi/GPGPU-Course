#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define MIN(a, b) ((a) < (b) ? a : b)

typedef struct argument {
  int *a;
  int *hist;
  int tid;
  int size;
  int hist_size;
  int num_thread;
} argument;

void validate(int *a, int *b, int length) {
  for (int i = 0; i < length; ++i) {
    if (a[i] != b[i]) {
      printf("Different value detected at position: %d,"
             " expected %d but get %d\n",
             i, a[i], b[i]);
      return;
    }
  }
  printf("There is no differences detected! All the same!\n");
}

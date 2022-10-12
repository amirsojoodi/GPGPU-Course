#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

void sequential_naive_vectorAdd(int *a, int *b, int *c, int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

void sequential_vectorAdd_first_optimization(int *a, int *b, int *c, int size) {
  // for(int i = 0; i < size / 2; i++){
  //	c[i] = a[i] + b[i];
  //	c[size - i - 1] = a[size - i - 1] + b[size - i - 1];
  //}
  for (int i = 0; i < size / 2;) {
    c[i] = a[i] + b[i];
    i++;
    c[size - i] = a[size - i] + b[size - i];
  }
}

void sequential_vectorAdd_second_optimization(int *a, int *b, int *c,
                                              int size) {
  for (int i = 0; i < size; i += 2) {
    c[i] = a[i] + b[i];
    c[i + 1] = a[i + 1] + b[i + 1];
  }
}

void sequential_vectorAdd_third_optimization(int *a, int *b, int *c, int size) {
  for (int i = 0, j = size / 2; i < size / 2; i++, j++) {
    c[i] = a[i] + b[i];
    c[j] = a[j] + b[j];
  }
}

void sequential_vectorAdd_forth_optimization(int *a, int *b, int *c, int size) {
  for (int i = 0; i < size; i += 8) {
    c[i] = a[i] + b[i];
    c[i + 1] = a[i + 1] + b[i + 1];
    c[i + 2] = a[i + 2] + b[i + 2];
    c[i + 3] = a[i + 3] + b[i + 3];
    c[i + 4] = a[i + 4] + b[i + 4];
    c[i + 5] = a[i + 5] + b[i + 5];
    c[i + 6] = a[i + 6] + b[i + 6];
    c[i + 7] = a[i + 7] + b[i + 7];
  }
}

void print_array(char *array_name, int *array, int size) {
  printf("\n%s: \n[", array_name);
  for (int i = 0; i < size; i++) {
    printf("%d, ", array[i]);
  }
  printf("]\n");
}

int main(int argc, char *argv[]) {

  int *a, *b, *c;
  struct timeval start, end;
  time_t t;

  if (argc != 2) {
    printf("Correct way to execute this program is:\n");
    printf("./vectorOp data_count\n");
    printf("For example:\n./vectorOp 10000 \n");
    return 1;
  }
  int size = atoi(argv[1]);

  a = (int *)malloc(sizeof(int) * size);
  b = (int *)malloc(sizeof(int) * size);
  c = (int *)malloc(sizeof(int) * size);

  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
    c[i] = 0;
  }

  // naive

  gettimeofday(&start, NULL);

  sequential_naive_vectorAdd(a, b, c, size);

  gettimeofday(&end, NULL);

  double diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Naive VectorAdd time calculation duration: %8.5fms\n", diff / 1000);

  // first
  gettimeofday(&start, NULL);

  sequential_vectorAdd_first_optimization(a, b, c, size);

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("First VectorAdd time calculation duration: %8.5fms\n", diff / 1000);

  // two
  gettimeofday(&start, NULL);

  sequential_vectorAdd_second_optimization(a, b, c, size);

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Second VectorAdd time calculation duration: %8.5fms\n", diff / 1000);

  // three
  gettimeofday(&start, NULL);

  sequential_vectorAdd_third_optimization(a, b, c, size);

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Third VectorAdd time calculation duration: %8.5fms\n", diff / 1000);

  // four
  gettimeofday(&start, NULL);

  sequential_vectorAdd_forth_optimization(a, b, c, size);

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Fourth VectorAdd time calculation duration: %8.5fms\n", diff / 1000);
  // TODO: validation
  // TODO: validation
  // TODO: validation
  // TODO: validation
  // TODO: validation
  // print_array("A", a, size);
  // print_array("B", b, size);
  // print_array("C", c, size);

  free(a);
  free(b);
  free(c);

  return 0;
}

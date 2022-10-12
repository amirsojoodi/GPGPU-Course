#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

typedef struct node {
  int a;
  int b[100];
} node;

void sequential_naive_vectorAdd(node *first, node *second, node *out,
                                int size) {
  for (int i = 0; i < size; i++) {
    out[i].a = first[i].a + second[i].a;
  }
}

void sequential_vectorAdd_first_optimization(node *first, node *second,
                                             node *out, int size) {
  for (int i = 0; i < size / 2; i++) {
    out[i].a = first[i].a + second[i].a;
    out[size - i - 1].a = first[size - i - 1].a + second[size - i - 1].a;
  }
}

void sequential_vectorAdd_second_optimization(node *first, node *second,
                                              node *out, int size) {
  for (int i = 0; i < size; i += 2) {
    out[i].a = first[i].a + second[i].a;
    out[i + 1].a = first[i + 1].a + second[i + 1].a;
  }
}

void sequential_vectorAdd_third_optimization(node *first, node *second,
                                             node *out, int size) {
  for (int i = 0, j = size / 2; i < size / 2; i++, j++) {
    out[i].a = first[i].a + second[i].a;
    out[j].a = first[j].a + second[j].a;
  }
}

void sequential_vectorAdd_fourth_optimization(node *first, node *second,
                                              node *out, int size) {
  for (int i = 0; i < size; i += 8) {
    out[i].a = first[i].a + second[i].a;
    out[i + 1].a = first[i + 1].a + second[i + 1].a;
    out[i + 2].a = first[i + 2].a + second[i + 2].a;
    out[i + 3].a = first[i + 3].a + second[i + 3].a;
    out[i + 4].a = first[i + 4].a + second[i + 4].a;
    out[i + 5].a = first[i + 5].a + second[i + 5].a;
    out[i + 6].a = first[i + 6].a + second[i + 6].a;
    out[i + 7].a = first[i + 7].a + second[i + 7].a;
  }
}

void print_array(char *array_name, node *array, int size) {
  printf("\n%s: \n[", array_name);
  for (int i = 0; i < size; i++) {
    printf("%d, ", array[i].a);
  }
  printf("]\n");
}

int main(int argc, char *argv[]) {

  node *first, *second, *out;
  struct timeval start, end;
  time_t t;

  if (argc != 2) {
    printf("Correct way to execute this program is:\n");
    printf("./vectorOp data_count\n");
    printf("For example:\n./vectorOp 10000 \n");
    return 1;
  }
  int size = atoi(argv[1]);

  first = (node *)malloc(sizeof(node) * size);
  second = (node *)malloc(sizeof(node) * size);
  out = (node *)malloc(sizeof(node) * size);

  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    first[i].a = rand() % 100;
    second[i].a = rand() % 100;
    out[i].a = 0;
  }

  // naive

  gettimeofday(&start, NULL);

  sequential_naive_vectorAdd(first, second, out, size);

  gettimeofday(&end, NULL);

  double diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Naive VectorAdd time calculation duration: %8.5fms\n", diff / 1000);

  // first
  gettimeofday(&start, NULL);

  sequential_vectorAdd_first_optimization(first, second, out, size);

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("First VectorAdd time calculation duration: %8.5fms\n", diff / 1000);

  // two
  gettimeofday(&start, NULL);

  sequential_vectorAdd_second_optimization(first, second, out, size);

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Second VectorAdd time calculation duration: %8.5fms\n", diff / 1000);

  // three
  gettimeofday(&start, NULL);

  sequential_vectorAdd_third_optimization(first, second, out, size);

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Third VectorAdd time calculation duration: %8.5fms\n", diff / 1000);

  // four
  gettimeofday(&start, NULL);

  sequential_vectorAdd_fourth_optimization(first, second, out, size);

  gettimeofday(&end, NULL);

  diff =
      (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

  printf("Fourth VectorAdd time calculation duration: %8.5fms\n", diff / 1000);

  // print_array("A", a, size);
  // print_array("B", b, size);
  // print_array("C", c, size);

  free(first);
  free(second);
  free(out);

  return 0;
}

#include "pi.h"

int main(int argc, char *argv[]) {

  double elapsed_time;
  double serialPI, parallelPI, step;

  if (argc != 3) {
    printf("Correct way to execute this program is:\n");
    printf("./pi stepNum numberOfThreads\n");
    return 1;
  }

  int num_steps = atoi(argv[1]);
  int num_thread = atoi(argv[2]);

  serialPI = 0;
  parallelPI = 0;

  // Sequential histogram
  set_clock();

  // COMPLETE HERE
  step = 1.0 / num_steps;
  double x = 0;
  double sum = 0;
  for (int i = 0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
  }
  serialPI = sum * step;

  elapsed_time = get_elapsed_time();

  printf("Naive PI calculation time: %.4fms\n", elapsed_time / 1000);

  // Openmp Parallel histogram
  set_clock();

  // COMPLETE HERE
  // omp_set_num_threads(num_thread);

#pragma omp parallel num_threads(num_thread)
  {
    double private_sum = 0.0;
    double x = 0;
#pragma omp for
    for (int i = 0; i < num_steps; i++) {
      x = (i + 0.5) * step;
      private_sum += 4.0 / (1.0 + x * x);
    }
#pragma omp atomic
    parallelPI += private_sum;
  }
  parallelPI *= step;

  elapsed_time = get_elapsed_time();

  printf("-> Openmp PI calculation time: %.4fms\n", elapsed_time / 1000);

#ifdef TEST
  validate(&serialPI, &parallelPI, 1);
#endif

  return 0;
}

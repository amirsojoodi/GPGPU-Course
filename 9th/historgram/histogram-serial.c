#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H
#include"histogram.h"
#endif

int *sequential_naive_histogram(int *a, int *serial_hist, int size, int hist_size){

	int bucketSize = RANGE / hist_size;

    for (int i = 0; i < size; i++) {
        serial_hist[a[i] / bucketSize]++;
    }
}


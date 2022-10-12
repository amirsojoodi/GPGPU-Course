#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<sys/time.h>

#define min(a,b) ((a) < (b) ? a : b)

#define TEST 1

typedef struct argument{
	int *a;
	int *b;
	int *c;
	int tid;
	int size;
	int num_threads;
} argument;

void validate(int *a, int *b, int length) {
	for (int i = 0; i < length; ++i) {
		if (a[i] != b[i]) {
			printf("Different value detected at position: %d,"
					" expected %d but get %d\n", i, a[i], b[i]);
			break;
		}
	}
}


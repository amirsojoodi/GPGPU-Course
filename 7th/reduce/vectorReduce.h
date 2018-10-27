#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<sys/time.h>

#define MIN(a,b) ((a) < (b) ? a : b)

typedef struct argument{
	// You should complete here
	int *a;
	int id;
	int size;
	int my_result;
	int num_thread;
} argument;

int validate(int *a, int *b, int length) {
	for (int i = 0; i < length; ++i) {
		if (a[i] != b[i]) {
			printf("Different value detected at position: %d,"
					"expected %d but get %d\n", i, a[i], b[i]);
			return 1;
		}
	}
	printf("Validation test passed successfully.\n");
	return 0;
}

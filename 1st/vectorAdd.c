#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<time.h>

void vectorAdd(int *a, int *b, int *c, int size){
	for(int i = 0; i < size; i++){
		c[i] = a[i] + b[i];
	}
}

void print_array(char *array_name, int *array, int size){
	printf("\n%s: \n[", array_name);
	for(int i = 0; i < size; i++){
		printf("%d, ", array[i]);
	}
	printf("]\n");
}

int main(int argc, char *argv[]){

	int *a, *b, *c;
	struct timeval start, end;
	time_t t;

	int size = atoi(argv[1]);

	a = (int *)malloc(sizeof(int) * size);
	b = (int *)malloc(sizeof(int) * size);
	c = (int *)malloc(sizeof(int) * size);

	srand((unsigned) time(&t));

	for(int i=0; i < size; i++){
		a[i] = rand() % 100;
		b[i] = rand() % 100;
		c[i] = 0;
	}

	// naive

	gettimeofday(&start, NULL); 

	vectorAdd(a, b, c, size);

	gettimeofday(&end, NULL); 
	
	double diff = (end.tv_sec - start.tv_sec) * 1000000.0 + 
		(end.tv_usec - start.tv_usec);
	
	printf("VectorAdd time calculation duration: %8.5fms\n", diff / 1000);
	
	//print_array("A", a, size);
	//print_array("B", b, size);
	//print_array("C", c, size);

	free(a);
	free(b);
	free(c);

	return 0;
}

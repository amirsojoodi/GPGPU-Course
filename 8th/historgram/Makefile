all: histogram.o histogramOptimized.o

histogram.o: histogram.c histogram.h
	gcc -std=c99 histogram.c -o histogram.o -lpthread -D TEST

histogramOptimized.o: histogram.c histogram.h
	gcc -std=c99 histogram.c -o histogramOptimized.o -lpthread -O3 

clean:
	rm *.o

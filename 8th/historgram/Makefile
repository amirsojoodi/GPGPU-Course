all: histogram histogramOptimized

histogram: histogram.c histogram.h
	gcc -std=c99 histogram.c -o histogram -lpthread -D TEST

histogramOptimized: histogram.c histogram.h
	gcc -std=c99 histogram.c -o histogramOptimized -lpthread -O3 

clean:
	rm histogram histogramOptimized

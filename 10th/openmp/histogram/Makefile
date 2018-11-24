options = -lpthread -D TEST -std=c99 -Wall -fopenmp
sources = histogram.c histogram.h helper-functions.c histogram-serial.c histogram-parallel-3.c histogram-openmp.c

all: histogram histogramOptimized

histogram: $(sources)
	gcc $(sources) -o histogram $(options)

histogramOptimized: $(sources)
	gcc $(sources) -o histogramOptimized $(options) -O3

clean:
	rm histogram histogramOptimized

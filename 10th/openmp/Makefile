options = -lpthread -D TEST -std=c99 -Wall
sources = histogram.c histogram.h helper-functions.c histogram-serial.c histogram-parallel-1.c histogram-parallel-2.c histogram-parallel-3.c

all: histogram histogramOptimized

histogram: $(sources)
	gcc $(sources) -o histogram $(options)

histogramOptimized: $(sources)
	gcc $(sources) -o histogramOptimized $(options) -O3

clean:
	rm histogram histogramOptimized

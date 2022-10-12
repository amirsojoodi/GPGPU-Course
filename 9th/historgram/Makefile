options = -lpthread -D TEST -std=c99 -Wall
sources = histogram.c histogram.h helper-functions.c histogram-serial.c \
	histogram-parallel-1.c histogram-parallel-2.c histogram-parallel-3.c

all: histogram.o histogramOptimized.o

histogram.o: $(sources)
	gcc $(sources) -o histogram.o $(options)

histogramOptimized.o: $(sources)
	gcc $(sources) -o histogramOptimized.o $(options) -O3

clean:
	rm *.o

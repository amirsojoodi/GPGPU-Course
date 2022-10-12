options = -D TEST -std=c99 -Wall -fopenmp
sources = pi.c pi.h helper-functions.c

all: pi.o piOptimized.o

pi.o: $(sources)
	gcc $(sources) -o pi.o $(options)

piOptimized.o: $(sources)
	gcc $(sources) -o piOptimized.o $(options) -O3

clean:
	rm *.o

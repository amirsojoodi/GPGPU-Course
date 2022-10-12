options = -D TEST -std=gnu99 -Wall -fopenmp
sources = pi.c pi.h helper-functions.c

all: pi.o pi_optimized.o

pi.o: $(sources)
	gcc $(sources) -o pi.o $(options)

pi_optimized.o: $(sources)
	gcc $(sources) -o pi_optimized.o $(options) -O3

clean:
	rm *.o

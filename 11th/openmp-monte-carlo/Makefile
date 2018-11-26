options = -D TEST -std=c99 -Wall -fopenmp
sources = pi.c pi.h helper-functions.c

all: pi pi_optimized

pi: $(sources)
	gcc $(sources) -o pi $(options)

pi_optimized: $(sources)
	gcc $(sources) -o pi_optimized $(options) -O3

clean:
	rm pi pi_optimized

options = -D TEST -std=c99 -Wall -fopenmp
sources = pi.c pi.h helper-functions.c

all: pi piOptimized

pi: $(sources)
	gcc $(sources) -o pi $(options)

piOptimized: $(sources)
	gcc $(sources) -o piOptimized $(options) -O3

clean:
	rm pi piOptimized

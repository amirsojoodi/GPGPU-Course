all: vectorReduce.o vectorReduceOptimized.o

vectorReduce.o: vectorReduce.c vectorReduce.h
	gcc -std=c99 vectorReduce.c -o vectorReduce.o -lpthread -D TEST

vectorReduceOptimized.o: vectorReduce.c vectorReduce.h
	gcc -std=c99 vectorReduce.c -o vectorReduceOptimized.o -O3 -lpthread -D TEST

clean:
	rm *.o

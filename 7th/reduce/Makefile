all: vectorReduce vectorReduceOptimized 

vectorReduce: vectorReduce.c vectorReduce.h
	gcc -std=c99 vectorReduce.c -o vectorReduce -lpthread -D TEST

vectorReduceOptimized: vectorReduce.c vectorReduce.h
	gcc -std=c99 vectorReduce.c -o vectorReduceOptimized -O3 -lpthread -D TEST

clean:
	rm vectorReduce vectorReduceOptimized

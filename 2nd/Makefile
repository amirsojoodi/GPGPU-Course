all: vectorAdd.o vectorAddOptimized.o

vectorAdd.o: vectorAdd.c
	gcc vectorAdd.c -o vectorAdd.o -Wall -std=c99

vectorAddOptimized.o: vectorAdd.c
	gcc vectorAdd.c -o vectorAddOptimized.o -Wall -std=c99 -O3

clean:
	rm *.o

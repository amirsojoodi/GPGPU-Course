all: vectorAdd.o vectorAddOptimized.o 

vectorAdd.o: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAdd.o -lpthread -lm

vectorAddOptimized.o: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAddOptimized.o -O3 -lpthread -lm

clean:
	rm *.o

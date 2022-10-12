all: vectorAdd.o vectorAddDebug.o

vectorAdd.o: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAdd.o -lpthread -lm

vectorAddDebug.o: vectorAdd.c 
	gcc -g -std=c99 vectorAdd.c -o vectorAddDebug.o -lpthread -lm
	echo "Use 'gdb vectorAddDebug.o' to debug"

clean:
	rm *.o

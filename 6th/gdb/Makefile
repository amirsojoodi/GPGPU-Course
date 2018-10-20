all: vectorAdd vectorAddDebug

vectorAdd: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAdd -lpthread -lm

vectorAddDebug: vectorAdd.c 
	gcc -g -std=c99 vectorAdd.c -o vectorAddDebug -lpthread -lm
	echo "Use 'gdb vectorAddDebug' to debug"

clean:
	rm vectorAdd vectorAddDebug

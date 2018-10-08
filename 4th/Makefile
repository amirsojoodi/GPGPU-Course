all: vectorAdd vectorAddOptimized 

vectorAdd: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAdd -lpthread -lm

vectorAddOptimized: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAddOptimized -O3 -lpthread -lm

clean:
	rm vectorAdd vectorAddOptimized

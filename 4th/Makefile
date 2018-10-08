all: vectorAdd vectorAddOptimized 

vectorAdd: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAdd -lpthread

vectorAddOptimized: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAddOptimized -O3 -lpthread 

clean:
	rm vectorAdd vectorAddOptimized

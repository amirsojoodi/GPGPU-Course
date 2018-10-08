all: vectorAdd vectorAddOptimized

vectorAdd: vectorAdd.c
	gcc vectorAdd.c -o vectorAdd -Wall -std=c99

vectorAddOptimized: vectorAdd.c
	gcc vectorAdd.c -o vectorAddOptimized -Wall -std=c99 -O3

clean:
	rm vectorAdd vectorAddOptimized 

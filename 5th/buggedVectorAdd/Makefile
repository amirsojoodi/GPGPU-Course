all: vectorAdd vectorAddOptimized 

vectorAdd: vectorAdd.c vectorAdd.h
	gcc -std=c99 vectorAdd.c -o vectorAdd -lpthread -lm -D TEST

vectorAddOptimized: vectorAdd.c vectorAdd.h
	gcc -std=c99 vectorAdd.c -o vectorAddOptimized -O3 -lpthread -lm 

clean:
	rm vectorAdd vectorAddOptimized

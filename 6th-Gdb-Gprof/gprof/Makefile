all: vectorAdd.o vectorAddProfiled.o

vectorAdd.o: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAdd.o -lpthread -lm

vectorAddProfiled.o: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAddProfiled.o -lpthread -lm -g -pg

analysis: vectorAddProfiled.o gmon.out
	gprof -b vectorAddProfiled.o gmon.out > analysis.out

clean:
	rm vectorAdd.o vectorAddProfiled.o gmon.out analysis.out

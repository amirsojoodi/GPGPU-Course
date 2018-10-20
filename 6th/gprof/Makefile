all: vectorAdd vectorAddProfiled

vectorAdd: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAdd -lpthread -lm

vectorAddProfiled: vectorAdd.c 
	gcc -std=c99 vectorAdd.c -o vectorAddProfiled -lpthread -lm -pg

analysis: vectorAddProfiled gmon.out
	gprof -b vectorAddProfiled gmon.out > analysis.out

clean:
	rm vectorAdd vectorAddProfiled gmon.out analysis.out

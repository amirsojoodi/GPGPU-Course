gcc-options = -D TEST -std=gnu99 -Wall 
nvcc-opetions = -Wall -D TEST -Wextra
sources = vectorOp.cu vectorOp.h helper-functions.o

all: vectorOp vectorOp_optimized

vectorOp: $(sources) Makefile
	nvcc $(sources) -o vectorOp $(nvcc-options)

vectorOp_optimized: $(sources) Makefile
	nvcc $(sources) -o vectorOp_optimized $(nvcc-options) -O3

helper_function.o: helper_function.c vectorOp.h Makefile
	gcc -c helper_function.c $(gcc-options)

clean:
	rm vectorOp vectorOp_optimized

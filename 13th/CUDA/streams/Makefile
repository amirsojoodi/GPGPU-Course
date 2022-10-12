nvcc_options= -lm -D TEST --compiler-options -Wall --default-stream per-thread 
sources = vectorOp.cu

all: vectorOp.o vectorOp_optimized.o

vectorOp.o: $(sources) Makefile vectorOp.h
	nvcc -o vectorOp.o $(sources) $(nvcc_options)

vectorOp_optimized.o: $(sources) Makefile vectorOp.h
	nvcc -o vectorOp_optimized.o $(sources) $(nvcc_options) -O3

clean:
	rm *.o

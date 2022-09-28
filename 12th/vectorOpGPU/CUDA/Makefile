nvcc_options= -lm -D TEST --compiler-options -Wall 
sources = vectorOp.cu

all: vectorOp vectorOp_optimized

vectorOp: $(sources) Makefile vectorOp.h
	nvcc -o vectorOp $(sources) $(nvcc_options)

vectorOp_optimized: $(sources) Makefile vectorOp.h
	nvcc -o vectorOp_optimized $(sources) $(nvcc_options) -O3

clean:
	rm vectorOp vectorOp_optimized

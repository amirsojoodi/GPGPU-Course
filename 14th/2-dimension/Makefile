nvcc_options= -lm -D TEST --compiler-options -Wall 
sources = blur.cu

all: blur.o blur_optimized.o

blur.o: $(sources) Makefile blur.h
	nvcc -o blur.o $(sources) $(nvcc_options)

blur_optimized.o: $(sources) Makefile blur.h
	nvcc -o blur_optimized.o $(sources) $(nvcc_options) -O3

clean:
	rm *.o

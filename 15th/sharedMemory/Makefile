nvcc_options= -gencode arch=compute_30,code=sm_30 -lm -D TEST --compiler-options -Wall 
sources = blur.cu

all: blur blur_optimized

blur: $(sources) Makefile blur.h
	nvcc -o blur $(sources) $(nvcc_options)

blur_optimized: $(sources) Makefile blur.h
	nvcc -o blur_optimized $(sources) $(nvcc_options)-O3

clean:
	rm blur blur_optimized

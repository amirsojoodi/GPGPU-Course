OPT  = -Ofast
STD  = c++17
WARN = -Wall -Wextra

VARS = -DTEST -DOPENCL_2

CXXSRC = main.cc
CXXHEAD = main.hxx

CXXFLAGS = $(WARN) -std=$(STD) $(VARS) -pipe `pkg-config --cflags OpenCL`
LDFLAGS  = -lpthread -lm `pkg-config --libs OpenCL`

all: default

default: vectorOp.o vectorOp_optimized.o

vectorOp.o: $(CXXSRC) $(CXXHEAD) Makefile
	$(CXX) -o vectorOp.o $(CXXSRC) $(CXXFLAGS) $(LDFLAGS)

vectorOp_optimized.o: $(CXXSRC) $(CXXHEAD) Makefile
	$(CXX) -o vectorOp_optimized.o $(CXXSRC) $(OPT) $(CXXFLAGS) $(LDFLAGS)

clean:
	$(RM) *.o


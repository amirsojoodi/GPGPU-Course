OPT  = -Ofast
STD  = c++17
WARN = -Wall -Wextra

VARS = -DTEST -DOPENCL_2

CXXSRC = main.cc
CXXHEAD = main.hxx

CXXFLAGS = $(WARN) -std=$(STD) $(VARS) -pipe `pkg-config --cflags OpenCL`
LDFLAGS  = -lpthread -lm `pkg-config --libs OpenCL`

all: default

default: multiCq.o multiCq_optimized.o

multiCq: $(CXXSRC) $(CXXHEAD) Makefile
	$(CXX) -o multiCq.o $(CXXSRC) $(CXXFLAGS) $(LDFLAGS)

multiCq_optimized: $(CXXSRC) $(CXXHEAD) Makefile
	$(CXX) -o multiCq_optimized.o $(CXXSRC) $(OPT) $(CXXFLAGS) $(LDFLAGS)

clean:
	$(RM) *.o

OPT  = -Ofast
STD  = c++17
WARN = -Wall -Wextra

VARS = -DTEST -DOPENCL_2

CXXSRC = main.cc
CXXHEAD = main.hxx

CXXFLAGS = $(WARN) -std=$(STD) $(VARS) -pipe `pkg-config --cflags OpenCL`
LDFLAGS  = -lpthread -lm `pkg-config --libs OpenCL`

all: default

default: multiCq multiCq_optimized

multiCq: $(CXXSRC) $(CXXHEAD) Makefile
	$(CXX) -o multiCq $(CXXSRC) $(CXXFLAGS) $(LDFLAGS)

multiCq_optimized: $(CXXSRC) $(CXXHEAD) Makefile
	$(CXX) -o multiCq_optimized $(CXXSRC) $(OPT) $(CXXFLAGS) $(LDFLAGS)

clean:
	$(RM) multiCq multiCq_optimized

OPT  = -Ofast
STD  = c++17
WARN = -Wall -Wextra

VARS = -DTEST -DOPENCL_2

CXXSRC = main.cc
CXXHEAD = main.hxx

CXXFLAGS = $(WARN) -std=$(STD) $(VARS) -pipe `pkg-config --cflags OpenCL`
LDFLAGS  = -lpthread -lm `pkg-config --libs OpenCL`

all: default

default: vectorOp vectorOp_optimized

vectorOp: $(CXXSRC) $(CXXHEAD) Makefile
	$(CXX) -o vectorOp $(CXXSRC) $(CXXFLAGS) $(LDFLAGS)

vectorOp_optimized: $(CXXSRC) $(CXXHEAD) Makefile
	$(CXX) -o vectorOp_optimized $(CXXSRC) $(OPT) $(CXXFLAGS) $(LDFLAGS)

clean:
	$(RM) vectorOp vectorOp_optimized


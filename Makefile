DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024

LOW ?=  0
HIGH ?= 2


CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -g -O3
CFLAGS += -DSURROGATES
NVCCFLAGS = --gpu-architecture=sm_35
LDFLAGS += -lm

all: gendata myknn

gendata: gendata.o
	nvcc -o gendata gendata.o $(LDFLAGS)

gendata.o: gendata.cu func.cu
	nvcc $(CFLAGS) -c gendata.cu

myknn: myknn.o
	nvcc -o myknn myknn.o $(LDFLAGS)

myknn.o: myknn.cu func.cu
	nvcc $(CFLAGS) $(NVCCFLAGS) -c myknn.cu

clean:
	rm -f myknn *.o gendata

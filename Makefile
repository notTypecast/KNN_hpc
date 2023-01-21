DIM ?= 16
KNN ?= 32
TRA ?= 262144
QUE ?= 1024

LOW ?=  0
HIGH ?= 2


CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -g -O3
CFLAGS += -DSURROGATES
NVCCFLAGS = -O3
LDFLAGS += -lm

all: gendata myknn

gendata: gendata.o
	gcc -o gendata gendata.o $(LDFLAGS)

gendata.o: gendata.c func.c
	gcc $(CFLAGS) -c gendata.c

myknn: myknn.o
	nvcc -o myknn myknn.o $(LDFLAGS)

myknn.o: myknn.cu func.c
	nvcc $(CFLAGS) $(NVCCFLAGS) -c myknn.cu

clean:
	rm -f myknn *.o gendata

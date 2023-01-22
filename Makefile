DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024

LOW ?=  0
HIGH ?= 2


CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -g -O3
CFLAGS += -DSURROGATES
PGCCFLAGS = -acc -fast -Minfo=accel
LDFLAGS += -lm

all: gendata myknn

gendata: gendata.o
	gcc -o gendata gendata.o $(LDFLAGS)

gendata.o: gendata.c func.c
	gcc $(CFLAGS) -c gendata.c

myknn: myknn.c func.c
	pgcc $(PGCCFLAGS) $(CFLAGS) myknn.c -o myknn $(LDFLAGS)

clean:
	rm -f myknn *.o gendata

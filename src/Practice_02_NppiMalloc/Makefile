CC=gcc
CUDA_PATH=/usr/local/cuda
CFLAGS= -g $(shell pkg-config --cflags libjpeg) -I$(CUDA_PATH)/include -I../utils
LDFLAGS= -pthread $(shell pkg-config --libs libjpeg) -L$(CUDA_PATH)/lib64 -lnppisu
SOURCES=main.c ../utils/jpeg_util.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=example

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.o:
	$(CC) $< $(CFLAGS) -c $@

clean:
	rm $(OBJECTS) $(EXECUTABLE)

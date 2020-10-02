CC=gcc
CUDA_PATH=/usr/local/cuda
CFLAGS= -g $(shell pkg-config --cflags libjpeg) -I$(CUDA_PATH)/include -I../src/utils
LDFLAGS= -pthread $(shell pkg-config --libs libjpeg) -L$(CUDA_PATH)/lib64 -lcuda -lcudart
SOURCES=create_rect_image.c ../src/utils/jpeg_util.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=create-rect-image

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.o:
	$(CC) $< $(CFLAGS) -c $@

clean:
	rm $(OBJECTS) $(EXECUTABLE)

CC=g++
LD=g++
CFLAGS=-g -Wall -Wno-sign-compare
LDFLAGS=-g -Wall
CAFFE_ROOT=/home/pengpeng/Ristretto/caffe/
INCLUDE_PATH=-I$(CAFFE_ROOT)include/ -I$(CAFFE_ROOT).build_release/src -I/usr/local/cuda/include -I/usr/include -I/usr/local/include
LIBS=-L$(CAFFE_ROOT)build/lib -lcaffe -lglog -lboost_system -lprotobuf -Wl,-R$(CAFFE_ROOT)build/lib -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

all:ConvertCaffemodel NetExport

ConvertCaffemodel:ConvertCaffemodel.o
	$(LD) -o $@ $? $(LDFLAGS) $(LIBS)

ConvertCaffemodel.o:ConvertCaffemodel.cpp
	$(CC) -c -o $@ $? $(CFLAGS) $(INCLUDE_PATH)

NetExport:NetExport.o
	$(LD) -o $@ $? $(LDFLAGS) $(LIBS)

NetExport.o:NetExport.cpp
	$(CC) -c -o $@ $? $(CFLAGS) $(INCLUDE_PATH)

clean:
	rm -rf *.o ConvertCaffemodel NetExport

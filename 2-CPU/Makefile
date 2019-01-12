CC = gcc
NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -Wall -O3 # -std=c11
CUDA_FLAGS = -O3
LIBS = -lm -L/usr/local/cuda/lib64/ -lcuda -lcudart

EXECS = neutron-seq neutron-openmp

H = 1.0
N = 500000000
Cc = 0.5
Cs = 0.5

OUTPUT = 0

all: $(EXECS)

# compiling rules
neutron-seq : neutron-seq.c
	$(CC) $(CFLAGS) -DOUTPUT=$(OUTPUT) $< -o $@ $(OBJECTS) $(LIBS)
	
neutron-openmp : neutron-openmp.c
	$(CC) $(CFLAGS) -DOUTPUT=$(OUTPUT) -fopenmp $< -o $@ $(OBJECTS) -lm

%.o : %.cu
	$(NVCC) -c $(CUDA_FLAGS) $<

# run simple execution
exec : all
	@echo ""
	@echo "EXECUTION..."
	@echo "============"
	@(for e in $(EXECS) ; do echo "Running: $$e"; echo "--------"; ./$$e $(H) $(N) $(Cc) $(Cs); echo "========"; echo ""; done)

clean:
	rm -f $(EXECS) *.o *~

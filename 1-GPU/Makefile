CC = gcc
NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -Wall -O3 # -std=c11
CUDA_FLAGS = -O3
LIBS = -lm -L/usr/local/cuda/lib64/ -lcuda -lcudart

EXECS = neutron-seq neutron-gpu_auto-batch neutron-gpu_manual-batch

H = 1.0
N = 500000000
Cc = 0.5
Cs = 0.5

OUTPUT = 0

all: $(EXECS)

# compiling rules
neutron-seq : neutron-seq.c
	$(CC) $(CFLAGS) -DOUTPUT=$(OUTPUT) $< -o $@ $(OBJECTS) $(LIBS)

neutron-gpu_auto-batch : neutron-gpu.o simulation_auto-batch.o
	$(CC) $(CFLAGS) -o $@ $+ $(LIBS)
neutron-gpu_manual-batch : neutron-gpu.o simulation_manual-batch.o
	$(CC) $(CFLAGS) -o $@ $+ $(LIBS)

neutron-gpu.o : neutron-gpu.c
	$(CC) -c $(CFLAGS) -DOUTPUT=$(OUTPUT) $<

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

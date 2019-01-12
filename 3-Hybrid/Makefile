CC = gcc
NVCC = /usr/local/cuda/bin/nvcc
MPICC = /home/sasl/shared/main/HPC/OpenMPI/bin/mpicc
CFLAGS = -Wall -O3 # -std=c11
CUDA_FLAGS = -O3
LIBS = -lm -L/usr/local/cuda/lib64/ -lcuda -lcudart

EXECS = neutron-seq neutron-hybrid_openmp neutron-hybrid_mpi

H = 1.0
N = 500000000
Cc = 0.5
Cs = 0.5

OUTPUT = 0

all: $(EXECS)

# compiling rules
neutron-seq : neutron-seq.c
	$(CC) $(CFLAGS) -DOUTPUT=$(OUTPUT) $< -o $@ $(OBJECTS) $(LIBS)
	
neutron-hybrid_openmp : neutron-hybrid_openmp.o simulation_auto-batch.o
	$(NVCC) $(CUDA_FLAGS) -o $@ $+ $(LIBS) -lgomp

neutron-hybrid_mpi : neutron-hybrid_mpi.o simulation_auto-batch.o
	$(MPICC) $(CFLAGS) -o $@ $+ $(LIBS) -L/home/sasl/shared/main/HPC/OpenMPI/lib/libmpi.so -lmpi

neutron-hybrid_openmp.o : neutron-hybrid_openmp.c
	$(NVCC) -Xcompiler -fopenmp -c $(CUDA_FLAGS) -DOUTPUT=$(OUTPUT) $<

neutron-hybrid_mpi.o : neutron-hybrid_mpi.c
	$(MPICC) -c $(CFLAGS) -DOUTPUT=$(OUTPUT) $<

%.o : %.cu
	$(NVCC) -c $(CUDA_FLAGS) $<

# run simple execution
exec : all
	@echo ""
	@echo "EXECUTION..."
	@echo "============"
	@echo "Running: neutron-seq"; echo "--------"
	@(./neutron-seq $(H) $(N) $(Cc) $(Cs); echo "========"; echo "")
	@echo "Running: neutron-hybrid_openmp"; echo "--------"
	@(./neutron-hybrid_openmp $(H) $(N) $(Cc) $(Cs); echo "========"; echo "")
	@echo "Running: neutron-hybrid_mpi"; echo "--------"
	@(/home/sasl/shared/main/HPC/OpenMPI/bin/./mpiexec -np 9 neutron-hybrid_mpi $(H) $(N) $(Cc) $(Cs); echo "========"; echo "")

clean:
	rm -f $(EXECS) *.o *~

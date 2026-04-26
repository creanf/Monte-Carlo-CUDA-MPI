SIMS ?= 8
DAYS ?= 4
RANKS ?= 1
	
create_sims: 
	nvcc -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_89,code=compute_89 \
     -O2 -c create_sims.cu -o kernels.o
	mpicc -I/usr/local/cuda/include -c start_sims.c -o main.o
	nvcc main.o kernels.o -o test -lmpi
	

run:
	mpirun -np $(RANKS) ./test 1 $(SIMS) $(DAYS)
CUDA_INC=/usr/local/cuda-11.0/targets/ppc64le-linux/include
MPI_INC=/opt/ibm/spectrum_mpi/include
INC =-I$(CUDA_INC) -I$(MPI_INC)
SIMS ?= 8
DAYS ?= 4
RANKS ?= 1
	
create_sims: 
	nvcc -gencode arch=compute_70,code=sm_70 \
		-gencode arch=compute_70,code=compute_70 \
     -O2 -c create_sims.cu -o kernels.o
	mpicc -I/usr/local/cuda/include -c start_sims.c -o main.o
	nvcc main.o kernels.o -o test -L$(MPI_ROOT)/lib -lmpi
	

run:
	mpirun -np $(RANKS) ./test 1 $(SIMS) $(DAYS)

create_sims_new:
	mpixlc -O3 $(INC) start_sims.c -c -o start_sims-xlc.o
	nvcc -O3 -arch=sm_70 $(INC) create_sims.cu -c -o create_sims-nvcc.o
	mpixlc -O3 $(INC) start_sims-xlc.o create_sims-nvcc.o -o test -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++

create_sims_new2:
	source /etc/profile.d/modules.sh && module load xl_r spectrum-mpi && module spider cuda && export PATH=/usr/local/cuda-11.0/bin:$$PATH && export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$$LD_LIBRARY_PATH && mpixlc -O3 $(INC) start_sims.c -c -o start_sims-xlc.o && nvcc -O3 -arch=sm_70 $(INC) create_sims.cu -c -o create_sims-nvcc.o && mpixlc -O3 $(INC) start_sims-xlc.o create_sims-nvcc.o -o test -L/usr/local/cuda-11.0/lib64/ -lcudadevrt -lcudart -lstdc++	

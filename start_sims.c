#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mpi.h>
#include "timing.h"

/* Shared structs */
struct sim {
    float max_val;
    float min_val;
    float curr_val;
};

struct sim_sum_t {
    float max_val;
    float min_val;
    float curr_val;
};

// IBM POWER9 System clock with 512MHZ resolution.
static __inline__ ticks getticks(void);


/* CUDA launcher */
void cuda_launcher(struct sim** seq,
                   struct sim_sum_t** sim_sum,
                   int num_blocks,
                   int num_threads,
                   int local_n,
                   unsigned long long seed,
                   int num_days,
                   float sigma,
                   float mu,
                   float time_delta,
                   int rank);

float phi(float x) {
    return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
}

float expected_max(float S0, float mu, float sigma, float T) {
    float mean = S0 * expf(mu * T);
    return mean * 2.0f * phi(-sigma * sqrtf(T));  // normcdff = Phi()
}

float expected_min(float S0, float mu, float sigma, float T) {
    float mean = S0 * expf(mu * T);
    return mean * 2.0f * phi(sigma* sqrtf(T));
 }

int main(int argc, char **argv)
{
    int rank, size;
    unsigned long long seed;
    int num_sims, num_days;
    ticks compute_start, compute_finish;
    ticks io_start, io_finish;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s seed log2_sims log2_days\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        seed = (unsigned long long)atoi(argv[1]);
        num_sims = 1 << atoi(argv[2]);
        num_days = 1 << atoi(argv[3]);
    }

    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_sims, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_days, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    printf("num ranks=%d",size);

    int local_n = num_sims / size;
 
    printf("Rank %d handling %d simulations\n", rank, local_n);

    if (local_n == 0) {
        MPI_Finalize();
        return 0;
    }

    float mu = 0.08f;
    float sigma = 0.15f;
   // float time_delta = 1.0f / (float)num_days;
float time_delta = 1.0f / 251;

    int num_threads = 256;
    int num_blocks = 256;

    struct sim *seq = NULL;
    struct sim_sum_t *sim_sum = NULL;

    compute_start = getticks();

    cuda_launcher(&seq,
        &sim_sum,
        num_blocks,
        num_threads,
        local_n,
        seed + rank,
        num_days,
        sigma,
        mu,
        time_delta,
        rank);

    compute_finish = getticks();
    double compute_time = (double)(compute_finish - compute_start)/ (double) 512000000.0 ;

    struct sim exp;
    struct sim dif;

    float trading_years = num_days/251.0;
    exp.curr_val = 100 * expf(mu * trading_years); //  
    exp.min_val = expected_min(100, mu, sigma, trading_years); 
    exp.max_val = expected_max(100, mu, sigma, trading_years);

    dif.curr_val = fabsf((exp.curr_val/(sim_sum->curr_val*1.0f) - 1) * 100);
    dif.min_val = fabsf((exp.min_val/(sim_sum->min_val*1.0f) - 1) * 100);
    dif.max_val = fabsf((exp.max_val/(sim_sum->max_val*1.0f) - 1) * 100);

    printf("Rank %d -> curr: %.4f max: %.4f min: %.4f\n",
        rank,
        sim_sum->curr_val,
        sim_sum->max_val,
        sim_sum->min_val);
    printf("Theoretical Expected Values: curr: %.4f max: %.4f min: %.4f: \n ",
		    exp.curr_val,
		    exp.min_val,
		    exp.max_val );
   printf("Percent Differences: curr: %.4f max: %.4f min: %.4f: \n ",
		   dif.curr_val,
		   dif.min_val,
		   dif.max_val);


    
    MPI_Offset offset = (MPI_Offset)(rank * sizeof(float) * 3 * local_n);


    MPI_Datatype MPI_SIM;

    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_SIM);
    MPI_Type_commit(&MPI_SIM);

    MPI_Barrier(MPI_COMM_WORLD);

    io_start = getticks();

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD,
                  "output.bin",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL,
                  &fh);

    
    MPI_File_write_at_all(fh, offset, seq, local_n, MPI_SIM, MPI_STATUS_IGNORE);
  
    MPI_File_close(&fh);

    MPI_Barrier(MPI_COMM_WORLD);

    io_finish = getticks();
    double io_time = (double)(io_finish - io_start) / (double) 512000000.0;
    
    if( rank == 0 )
    {
        printf("num days: %d\n", num_days);
	printf("TOTAL TIME: %f\n", compute_time);
        printf("PRINT TO FILE TIME: %f\n", io_time);
    }

    //Free using Cudaruntime so can access values from cuda in main
    cudaFree(seq);
    cudaFree(sim_sum);

    MPI_Finalize();
    return 0;
}


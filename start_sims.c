#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
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
                   float time_delta);

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


    int local_n = num_sims / size;

    printf("Rank %d handling %d simulations\n", rank, local_n);

    if (local_n == 0) {
        MPI_Finalize();
        return 0;
    }

    float mu = 0.08f;
    float sigma = 0.15f;
    float time_delta = 1.0f / (float)num_days;

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

    printf("Rank %d -> curr: %f max: %f min: %f\n",
        rank,
        sim_sum->curr_val,
        sim_sum->max_val,
        sim_sum->min_val);


    
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
        printf("TOTAL TIME: %f\n", compute_time);
        printf("PRINT TO FILE TIME: %f\n", io_time);
    }

    //Free using Cudaruntime so can access values from cuda in main
    cudaFree(seq);
    cudaFree(sim_sum);


    MPI_Finalize();
    return 0;
}


// #include <stdio.h>
// #include <stdlib.h>
// #include <mpi.h>

// /* C-style structs shared with CUDA */
// struct sim {
//     float max_val;
//     float min_val;
//     float curr_val;
// };

// struct sim_sum_t {
//     double max_val;
//     double min_val;
//     double curr_val;
// };

// /* CUDA function (C ABI) */
// void cuda_launcher(struct sim** seq,
//                    struct sim_sum_t** sim_sum,
//                    int num_blocks,
//                    int num_threads,
//                    int local_n,
//                    unsigned long long seed,
//                    int num_days,
//                    float sigma,
//                    float mu,
//                    float time_delta);

// int main(int argc, char **argv)
// {
//     int rank, size;
//     unsigned long long seed;
//     int num_sims, num_days;

//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0) {
//         if (argc != 4) {
//             fprintf(stderr, "Usage: %s seed log2_sims log2_days\n", argv[0]);
//             MPI_Abort(MPI_COMM_WORLD, 1);
//         }

//         seed = (unsigned long long)atoi(argv[1]);
//         num_sims = 1 << atoi(argv[2]);
//         num_days = 1 << atoi(argv[3]);
//     }

//     MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&num_sims, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&num_days, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     int local_n = num_sims / size;

//     float mu = 0.08f;
//     float sigma = 0.15f;
//     float time_delta = 1.0f / (float)num_days;

//     int num_threads = 256;
//     int num_blocks = 256;

//     struct sim *seq = NULL;
//     struct sim_sum_t *sim_sum = NULL;

//     cuda_launcher(&seq,
//                   &sim_sum,
//                   num_blocks,
//                   num_threads,
//                   local_n,
//                   seed + rank,
//                   num_days,
//                   sigma,
//                   mu,
//                   time_delta);

//     printf("Rank %d -> curr: %f max: %f min: %f\n",
//            rank,
//            sim_sum->curr_val,
//            sim_sum->max_val,
//            sim_sum->min_val);

//     MPI_Finalize();
//     return 0;
// }


// #include <stdio.h>
// #include <stdlib.h>
// #include <mpi.h>


// struct sim;
// struct sim_sum_t;

// struct sim {
//     float max_val;
//     float min_val;
//     float curr_val; // cuda malloc does this anyway
// };

// struct sim_sum_t {
//         double max_val;
//         double min_val;
//         double curr_val;
// };

// void cuda_launcher(struct sim** seq,
//                    struct sim_sum_t** sim_sum,
//                    int num_blocks,
//                    int num_threads,
//                    int local_n,
//                    unsigned long long seed,
//                    int num_sims,
//                    int num_days,
//                    float sigma,
//                    float mu,
//                    float time_delta);

// int main(int argc, char ** argv){

//     int rank, size;
//     unsigned long long seed;
//     int num_sims, num_days;
//     int num_gpus;

//     MPI_Init(&argc, &argv);

//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
    
  

    

//     if(rank == 0){

//         if(argc != 4){ fprintf(stderr, "EXPECTED 4 ARGUMENTS BUT GOT %d", argc); MPI_Abort(MPI_COMM_WORLD, 1);}

//         seed = (unsigned long long) atoi(argv[1]); 
//         num_sims = 1<<atoi(argv[2]);
//         num_days = 1<<atoi(argv[3]);

//     }

//     //send argv variables to all ranks so can work in parallel
//     MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&num_sims, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&num_days, 1, MPI_INT, 0, MPI_COMM_WORLD);

//    // cudaGetDeviceCount(&num_gpus);
//     MPI_Barrier(MPI_COMM_WORLD);


//     //int gpu_id = rank % num_gpus;
//     // cudaSetDevice(gpu_id);

//     //assumes num_sims and size are even
//     int local_n = num_sims / size;
//     int start = rank*local_n;
//     float mu = 0.08f; // 8% expected annual return, typical of the US stock market
//     float sigma = 0.15f; // 15% stdev/volatility at end of year
//     //float num_days = 252.0f; // one year of trading days
//     float time_delta = 1.0f/num_days;
//     int num_threads = 256; //max is 1024
//     int num_blocks = 256; //max is 16384 or something
//     int total_threads = num_threads * num_blocks; // AKA total number of simulations
//     struct sim* seq;
//     struct sim_sum_t* sim_sum;

//     cuda_launcher(&seq, &sim_sum, num_blocks, num_threads, local_n, seed, num_sims, num_days, sigma, mu, time_delta);
//     // cudaMallocManaged(&sim_sum, (size_t)sizeof(sim));
//     // cudaMallocManaged(&seq, (size_t)(local_n * sizeof(sim)));
//     // create_random_sequence<<<num_blocks, num_threads>>> (seed, num_sims, seq, mu, sigma, num_days, time_delta);

//     // cudaDeviceSynchronize();


//     // avg_sims(seq, num_sims, sim_sum, num_blocks, num_threads);

//     fprintf(stderr, "curr: %f     max: %f      min: %f     \n", sim_sum->curr_val, sim_sum->max_val, sim_sum->min_val);

//     // get sub arrays






//     // //print sub arrays to file
    
//     // char a;

//     // if(rank == 0){
//     //     a = 'a';
//     // }else{
//     //     a = 'c';
//     // }

//     // MPI_Offset offset = (MPI_Offset)rank *2;


//     // MPI_File fh;
//     // MPI_File_open(MPI_COMM_WORLD,
//     //               "output.txt",
//     //               MPI_MODE_CREATE | MPI_MODE_WRONLY,
//     //               MPI_INFO_NULL,
//     //               &fh);

//     // MPI_File_write_at_all(fh, offset, &a, 1, MPI_BYTE, MPI_STATUS_IGNORE);
  
//     // MPI_File_close(&fh);

//     // MPI_Barrier(MPI_COMM_WORLD);

    

//     //div = arrlen / size
//     //do sim on (rank*div, rank*div + div)


//     //if rnak = 0 output array





//     MPI_Finalize();
//     return EXIT_SUCCESS;


// }
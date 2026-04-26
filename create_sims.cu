#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
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


/* ================= KERNEL ================= */

__global__ void create_random_sequence(unsigned long long seed,
                                       int num_sims,
                                       struct sim* output,
                                       float mu,
                                       float sigma,
                                       int num_days,
                                       float time_delta)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_sims; i += stride) {

        curandStatePhilox4_32_10_t state;
        curand_init(seed, i, 0, &state);

        float curr = 100.0f;
        float max_v = 100.0f;
        float min_v = 100.0f;

        for (int j = 0; j < num_days; j++) {
            float z = curand_normal(&state);

            float growth =
                expf((mu - 0.5f * sigma * sigma) * time_delta +
                     sigma * sqrtf(time_delta) * z);

            curr *= growth;

            if (curr > max_v) max_v = curr;
            if (curr < min_v) min_v = curr;
        }

        output[i].curr_val = curr;
        output[i].max_val = max_v;
        output[i].min_val = min_v;
    }
}

/* ================= REDUCTION ================= */

__global__ void sum_sims(struct sim* sims,
                         int n,
                         struct sim_sum_t* out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        atomicAdd(&out->max_val, sims[i].max_val);
        atomicAdd(&out->min_val, sims[i].min_val);
        atomicAdd(&out->curr_val, sims[i].curr_val);
    }
}

/* ================= WRAPPER ================= */

extern "C"
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
                   int rank)
{   

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int dev = rank % num_gpus;
    cudaSetDevice(dev);

    cudaMallocManaged(seq, local_n * sizeof(struct sim));
    cudaMallocManaged(sim_sum, sizeof(struct sim_sum_t));

    (*sim_sum)->max_val = 0.0f;
    (*sim_sum)->min_val = 0.0f;
    (*sim_sum)->curr_val = 0.0f;

    create_random_sequence<<<num_blocks, num_threads>>>(
        seed,
        local_n,
        *seq,
        mu,
        sigma,
        num_days,
        time_delta
    );
    cudaDeviceSynchronize();

    sum_sims<<<num_blocks, num_threads>>>(*seq, local_n, *sim_sum);


    cudaDeviceSynchronize();

    (*sim_sum)->max_val /= local_n;
    (*sim_sum)->min_val /= local_n;
    (*sim_sum)->curr_val /= local_n;

    
    cudaDeviceSynchronize();
    
}
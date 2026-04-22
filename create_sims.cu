#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>

struct sim {
    float max_val = 0.0f;
    float min_val = 0.0f;
    float curr_val = 0.0f; // cuda malloc does this anyway
};

struct sim_sum_t {
        double max_val = 0;
        double min_val = 0;
        double curr_val = 0;
};


__global__ void create_random_sequence(unsigned long long seed, int num_sims, sim* output, float mu, float sigma, float num_days, float time_delta) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    for(int i = tid; i < num_sims; i+= grid_size){

        curandStatePhilox4_32_10_t state;
        curand_init(seed, i , 0, &state);
        output[i].curr_val = 100;

        for (int j = 0; j < num_days; ++j) {
                float z = curand_normal(&state); // normal dist [0, 1) - random noise
                output[i].curr_val *= expf((mu - 0.5f*sigma*sigma)*time_delta + sigma*sqrtf(time_delta)*z);
                if (output[i].curr_val > output[i].max_val) {
                    output[i].max_val = output[i].curr_val;
                }
                if (output[i].curr_val < output[i].min_val || j == 0) {
                    output[i].min_val = output[i].curr_val;
                }
         }
    }
}


__global__ void sum_sims(sim* sims, int arr_size, sim_sum_t* sim_sum) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    for(int i = tid; i < arr_size; i+= grid_size){

        //atomicAdd neded instead of normal for thread safety (otherwise threads overwrite)
        atomicAdd(&(sim_sum->max_val), (double)sims[i].max_val);
        atomicAdd(&(sim_sum->min_val), (double)sims[i].min_val);
        atomicAdd(&(sim_sum->curr_val), (double)sims[i].curr_val);
    }
}

void avg_sims(sim* sims, int arr_size, sim_sum_t* sim_sum, int num_blocks, int num_threads){

    sim_sum->max_val = 0;
    sim_sum->min_val = 0;
    sim_sum->curr_val = 0;

    sum_sims<<<num_blocks, num_threads>>>(sims, arr_size, sim_sum);
    cudaDeviceSynchronize();


    sim_sum->max_val /= arr_size;
    sim_sum->min_val /= arr_size;
    sim_sum->curr_val /= arr_size;

}

int main(int argc, char * argv[]) {

    // hardcode a seed for now
    unsigned long long seed = (unsigned long long) atoi(argv[1]); // i.e. 42ULL

    sim* seq; // array with each of the simulations. Sim is a struct with current value, max value, and min value
    sim_sum_t* sim_sum;

    float mu = 0.08f; // 8% expected annual return, typical of the US stock market
    float sigma = 0.15f; // 15% stdev/volatility at end of year
    float num_days = 252.0f; // one year of trading days
    float time_delta = 1.0f/num_days;
    int num_sims = atoi(argv[2]);


    int num_threads = 256; //max is 1024
    int num_blocks = 256; //max is 16384 or something
    int total_threads = num_threads * num_blocks; // AKA total number of simulations
    cudaMallocManaged(&seq, (size_t)num_sims*sizeof(sim));
    cudaMallocManaged(&sim_sum, (size_t)sizeof(sim));


    create_random_sequence<<<num_blocks, num_threads>>> (seed, num_sims, seq, mu, sigma, num_days, time_delta);

    cudaDeviceSynchronize();


    avg_sims(seq, num_sims, sim_sum, num_blocks, num_threads);

    printf("curr: %f     max: %f      min: %f     \n", sim_sum->curr_val, sim_sum->max_val, sim_sum->min_val);


    cudaFree(sim_sum);
    cudaFree(seq);

    return 0;
}
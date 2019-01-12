/**
 * Programmation GPU 
 * Universite Pierre et Marie Curie
 * Simulation de transport de neutrons.
 *
 * M. Pecheux - Automne 2018
 * [MAIN5 - HPCA]
 */

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

extern "C" double my_gettimeofday();

#define NB_BLOCKS  256
#define NB_THREADS 256

/** 
 * Controle des erreurs CUDA et debugging. 
 */

#ifdef CUDA_DEBUG
#define CUDA_SYNC_ERROR() {						\
    cudaError_t sync_error;						\
    cudaDeviceSynchronize();						\
    sync_error = cudaGetLastError();					\
    if(sync_error != cudaSuccess) {					\
      fprintf(stderr, "[CUDA SYNC ERROR at %s:%d -> %s]\n",		\
	      __FILE__ , __LINE__, cudaGetErrorString(sync_error));	\
      exit(EXIT_FAILURE);						\
    }									\
  }
#else /* #ifdef CUDA_DEBUG */
#define CUDA_SYNC_ERROR()
#endif /* #ifdef CUDA_DEBUG */

#define CUDA_ERROR(cuda_call) {					\
    cudaError_t error = cuda_call;				\
    if(error != cudaSuccess){					\
      fprintf(stderr, "[CUDA ERROR at %s:%d -> %s]\n",		\
	      __FILE__ , __LINE__, cudaGetErrorString(error));	\
      exit(EXIT_FAILURE);					\
    }								\
    CUDA_SYNC_ERROR();						\
  }

/*
 * Generates a random number for the given thread.
 */
 
__device__ float uniform_random_number(curandState* global_state, int thread_id) {
    curandState local_state = global_state[thread_id];
    float RANDOM            = curand_uniform(&local_state);
    global_state[thread_id] = local_state;
    return RANDOM;
}

__global__ void setup_kernel_seeds(curandState* state, unsigned long seed) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void kernel_simulation(float c, float c_c, float c_s, float h, int n,
                                  int* res, float* absorbed,
                                  curandState* global_state) {
  int i  = blockDim.x * blockIdx.x + threadIdx.x;
  int gi = i;
  int idx;
   
  float d; // direction du neutron (0 <= d <= PI)
  float x; // position de la particule (0 <= x <= h)
  float L; // distance parcourue par le neutron avant la collision
  float u; // variable aléatoire uniforme
  
  // memoire partagee entre les threads du bloc
  __shared__ int R[NB_THREADS];
  __shared__ int B[NB_THREADS];
  __shared__ int T[NB_THREADS];
  R[threadIdx.x] = 0;
  B[threadIdx.x] = 0;
  T[threadIdx.x] = 0;
  
  while(i < n) {
    d = 0.0; x = 0.0;
    while (1) {
      u = uniform_random_number(global_state, gi);
      L = -(1 / c) * log(u);
      x = x + L * cos(d);
      if (x < 0) {
      	R[threadIdx.x] = R[threadIdx.x] + 1; // ajout d'1 neutron reflechi
      	break;
      } else if (x >= h) {
      	T[threadIdx.x] = T[threadIdx.x] + 1; // ajout d'1 neutron transmis
      	break;
      } else if ((u = uniform_random_number(global_state, gi)) < c_c / c) {
      	B[threadIdx.x] = B[threadIdx.x] + 1; // ajout d'1 neutron absorbe
        idx = atomicAdd(res+3, 1); // recuperation + incrementation du compteur
                                   // de neutrons absorbes
      	absorbed[idx] = x;  // stockage de la position d'absorption
      	break;
      } else {
      	u = uniform_random_number(global_state, gi);
      	d = u * M_PI;
      }
    }
    i += gridDim.x * blockDim.x; // saut d'un bloc
  }

  /* REDUCTION */
  // synchronisation des threads du bloc
  __syncthreads();
  // calcul et mise a jour des compteurs globaux
  int j = blockDim.x / 2;
  while (j > 0) {
    if (threadIdx.x < j) {
      R[threadIdx.x] += R[threadIdx.x + j];
      B[threadIdx.x] += B[threadIdx.x + j];
      T[threadIdx.x] += T[threadIdx.x + j];
    }
    j /= 2;
    __syncthreads();
  }

  // ajout du bloc par le premier GPU du bloc
  if (threadIdx.x == 0) {
    atomicAdd(res,   R[0]);
    atomicAdd(res+1, B[0]);
    atomicAdd(res+2, T[0]);
  }
}

/**
 * Effectue la simulation de 'n' particules.
 */

extern "C"
void gpu_simulation(float c, float c_c, float c_s, float h, int n, int** res,
                    float** absorbed) {
  
  /* Variables liees au chronometrage */
  double debut, fin;

  /* GPU allocation */
  int s_res = 4 * sizeof(int);
  int s_abs = n * sizeof(float);
  int   *d_res;
  float *d_abs;
  CUDA_ERROR(cudaMalloc((void**) &d_res, s_res));
  CUDA_ERROR(cudaMalloc((void**) &d_abs, s_abs));

  /* CPU > GPU transfers (synchronous) */
  CUDA_ERROR(cudaMemcpy(d_res, *res,      s_res, cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(d_abs, *absorbed, s_abs, cudaMemcpyHostToDevice));
  
  /* definition de la taille de la grille de GPUs */
  dim3 nbBlocks, nbThreads;
  nbThreads.x = NB_THREADS;
  nbThreads.y = nbThreads.z = 1;
  nbBlocks.x  = NB_BLOCKS;
  nbBlocks.y  = nbBlocks.z  = 1;

  printf("--------\n");
  printf("Nb blocs : %d\t\tNb threads par bloc : %d\n", nbBlocks.x, nbThreads.x);
  printf("[GPU] Taille de batch automatique.\n");

  /* variables pour la generation de nombres aleatoires */
  curandState* dev_states;
  CUDA_ERROR(cudaMalloc((void**) &dev_states, nbThreads.x*nbBlocks.x*sizeof(curandState)));

  /* preparation des graines aleatoires */
  setup_kernel_seeds <<<nbBlocks, nbThreads>>>(dev_states, unsigned(time(NULL)));

  /* debut du chronometrage */
  debut = my_gettimeofday();            

  /* lancement des kernels */
  kernel_simulation<<<nbBlocks, nbThreads>>>(c, c_c, c_s, h, n, d_res, d_abs, dev_states);

  /* GPU > CPU transfers (synchronous) */
  CUDA_ERROR(cudaMemcpy(*res,      d_res, s_res, cudaMemcpyDeviceToHost));
  CUDA_ERROR(cudaMemcpy(*absorbed, d_abs, s_abs, cudaMemcpyDeviceToHost));

  /* fin du chronometrage */
  fin = my_gettimeofday();
  printf("[GPU] Temps de calcul seul : %.10f seconde(s)\n", fin - debut);

  /* liberation memoire */
  CUDA_ERROR(cudaFree(d_res));
  CUDA_ERROR(cudaFree(d_abs));
  CUDA_ERROR(cudaFree(dev_states));
}

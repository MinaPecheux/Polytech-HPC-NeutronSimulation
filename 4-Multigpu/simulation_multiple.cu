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
                    float** absorbed, int nb_gpus) {
  int i, dev, pos;

  /* Variables liees au chronometrage */
  double debut, fin;

  /* allocation GPU (max 3 gpus) */
  int s_res = 4 * sizeof(int);
  int         *d_res[3];
  float       *d_abs[3];
  curandState *dev_states[3];

  /* Debug du nombre max de GPUs */
  int gpu_count;
  cudaGetDeviceCount(&gpu_count);
  printf("--------\n");
  printf("Nombre max de GPUs:    %d\n", gpu_count);
  printf("Nombre actuel de GPUs: %d\n", nb_gpus);

  /* definition de la taille de la grille de GPUs */
  dim3 nbBlocks, nbThreads;
  nbThreads.x = NB_THREADS;
  nbThreads.y = nbThreads.z = 1;
  nbBlocks.x  = NB_BLOCKS;
  nbBlocks.y  = nbBlocks.z  = 1;
  int dim     = nbThreads.x*nbBlocks.x;

  /* calcul du nombre de neutrons a traiter par chaque GPU */
  int *nb_neutrons = (int*) calloc(nb_gpus, sizeof(int));
  int tmp = 0;
  for (dev = 0; dev < nb_gpus; dev++) {
    nb_neutrons[dev] = n / nb_gpus;
    tmp += nb_neutrons[dev];
  }
  nb_neutrons[nb_gpus - 1] += n - tmp;

  printf("\nNombre de neutrons pour chaque GPU :\n");
  for (dev = 0; dev < nb_gpus; dev++)
    printf("GPU #%d : %d\n", dev, nb_neutrons[dev]);

  /* transferts CPU > GPU (asynchrones) */
  for (dev = 0; dev < nb_gpus; dev++) {
    cudaSetDevice(dev);
    CUDA_ERROR(cudaMalloc((void**) &(d_res[dev]), s_res));
    CUDA_ERROR(cudaMalloc((void**) &(d_abs[dev]), nb_neutrons[dev] * sizeof(float)));
    CUDA_ERROR(cudaMalloc((void**) &(dev_states[dev]), dim*sizeof(curandState)));
    CUDA_ERROR(cudaMemcpyAsync(d_res[dev], *res, s_res, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpyAsync(d_abs[dev], *absorbed,
                          nb_neutrons[dev]*sizeof(float), cudaMemcpyHostToDevice));
  }

  printf("\nNb blocs : %d\t\tNb threads par bloc : %d\n", nbBlocks.x, nbThreads.x);

  /* preparation des graines aleatoires */
  for (dev = 0; dev < nb_gpus; dev++)
    setup_kernel_seeds <<<nbBlocks, nbThreads>>>(dev_states[dev], unsigned(time(NULL)));
  
  /* debut du chronometrage */
  debut = my_gettimeofday();         
  
  /* lancement des kernels */
  for (dev = 0; dev < nb_gpus; dev++) {
    cudaSetDevice(dev);
    kernel_simulation<<<nbBlocks, nbThreads>>>
      (c, c_c, c_s, h, nb_neutrons[dev], d_res[dev], d_abs[dev], dev_states[dev]);
  }
  
  /* recuperation des resultats sur le GPU 0
     + transferts GPU > CPU (asychrones) */
  pos = 0;
  for (dev = 0; dev < nb_gpus; dev++) {
    cudaSetDevice(dev);
    if (dev > 0) {
      for (i = 0; i < 4; i++) (d_res[0])[i] += (d_res[dev])[i];
    }
  
    CUDA_ERROR(cudaMemcpyAsync(*absorbed+pos, d_abs[dev],
                          nb_neutrons[dev]*sizeof(float), cudaMemcpyDeviceToHost));
    pos += nb_neutrons[dev];
  }
  cudaSetDevice(0);
  CUDA_ERROR(cudaMemcpyAsync(*res, d_res[0], s_res, cudaMemcpyDeviceToHost));
  
  /* fin du chronometrage */
  fin = my_gettimeofday();
  printf("[GPU] Temps de calcul seul : %.10f seconde(s)\n", fin - debut);

  /* liberation memoire */
  free(nb_neutrons);
  for (dev = 0; dev < nb_gpus; dev++) {
    CUDA_ERROR(cudaFree(d_res[dev]));
    CUDA_ERROR(cudaFree(d_abs[dev]));
    CUDA_ERROR(cudaFree(dev_states[dev]));
  }
}

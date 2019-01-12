/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 *
 * M. Pecheux - Automne 2018
 * [MAIN5 - HPCA]
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define N               2000
#define OUTPUT_FILE     "/tmp/absorbed.dat"

char info[] = "\
Usage:\n\
    neutron-hybrid_openmp H Nb C_c C_s nb_threads gpu_cpu_ratio\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
    nb_threads: nombre de thread OMP\n\
    gpu_cpu_ratio: ratio d'equilibrage GPU/CPU (0.0: only GPU)\n\
\n\
Exemple d'execution : \n\
    neutron-hybrid_openmp 1.0 500000000 0.5 0.5 8\n\
";

/*
 * générateur uniforme de nombres aléatoires dans l'intervalle [0,1)
 */
struct drand48_data alea_buffer;

void init_uniform_random_number(unsigned int seed) {
  srand48_r(seed, &alea_buffer);
}

float uniform_random_number() {
  double res = 0.0; 
  drand48_r(&alea_buffer, &res);
  return res;
}

/*
 * notre gettimeofday()
 */
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

/*
 * Fonction de lancement des kernels
 */

extern float *gpu_simulation(float c, float c_c, float c_s, float h, int n,
                             int** res, float** absorbed);

/*
 * main()
 */
int main(int argc, char *argv[]) {
  // La distance moyenne entre les interactions neutron/atome est 1/c. 
  // c_c et c_s sont les composantes absorbantes et diffusantes de c. 
  float c, c_c, c_s;
  // épaisseur de la plaque
  float h;
  // distance parcourue par le neutron avant la collision
  float L;
  // direction du neutron (0 <= d <= PI)
  float d;
  // variable aléatoire uniforme
  float u;
  // position de la particule (0 <= x <= h)
  float x;
  // nombre d'échantillons
  int n, n_omp, n_gpu;
  // nombre de neutrons refléchis, absorbés et transmis
  int r, b, t;
  // chronometrage
  double start, finish;
  int i, j = 0; // compteurs
  // nombre de threads OMP
  int nb_threads;
  double gpu_cpu_ratio;

  if(argc == 1)
    fprintf( stderr, "%s\n", info);

  // valeurs par defaut
  h = 1.0;
  n = 500000000;
  c_c = 0.5;
  c_s = 0.5;
  nb_threads = 8;
  gpu_cpu_ratio = 0.0;

  // recuperation des parametres
  if (argc > 1)
    h = atof(argv[1]);
  if (argc > 2)
    n = atoi(argv[2]);
  if (argc > 3)
    c_c = atof(argv[3]);
  if (argc > 4)
    c_s = atof(argv[4]);
  if (argc > 5)
    nb_threads = atof(argv[5]);
  if (argc > 5)
    gpu_cpu_ratio = atof(argv[6]);
  r = b = t = 0;
  c = c_c + c_s;

  /* Verification de la taille de batch */
  if (n % N != 0) {
      printf("Erreur : le nombre d'échantillons doit être un multiple de N (N = %d, n = %d).\n", N, n);
      exit(EXIT_FAILURE);
  }

  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);

  n_omp = (int)(gpu_cpu_ratio * (float)n);
  n_gpu = n - n_omp;
  printf("\nRatio d'equilibrage OpenMP / GPU : %g\n", gpu_cpu_ratio);
  printf("Nombre d'echantillons OpenMP : %d\n", n_omp);
  printf("Nombre d'echantillons GPU    : %d\n", n_gpu);

  float *absorbed     = (float *) calloc(n, sizeof(float));
  // contient : r, b, t, id du dernier neutron absorbe
  int   *res_gpu      = (int *)   calloc(4, sizeof(int));
  float *absorbed_gpu = (float *) calloc(n_gpu, sizeof(float));
  
  // debut du chronometrage
  start = my_gettimeofday();
  
  // definition du nombre total de threads
  omp_set_num_threads(nb_threads);
    
  /* DEBUT DE LA SECTION PARALLELE */
  /* Probleme de serialisation : le thread master est attendu par les autres
   * threads. Avec un if/else, on a un blocage infini des autres threads...
   * (Donc ici on attend plutot la fin du traitement GPU pour ensuite enchainer
   *  sur le traitement CPU, meme si les traitements ne sont donc pas parallelises.
   *  On evite le blocage infini.) */
  #pragma omp parallel private(d, x, u, L, i) reduction(+:b,r,t)
  {
    // thread master gere le GPU
    if (omp_get_thread_num() == 0) {
      printf("[CPU] Nombre de threads OpenMP: %d.\n", omp_get_num_threads());
      if (n_gpu > 0) {
        // simulation
        gpu_simulation(c, c_c, c_s, h, n_gpu, &res_gpu, &absorbed_gpu);
        // recuperation des resultats (nb de neutrons reflechis, absorbes et transmis)
        r = res_gpu[0]; b = res_gpu[1]; t = res_gpu[2];
      }
    }
    // autres threads traitent des neutrons
    // init du generateur avec le numero de thread
    init_uniform_random_number(omp_get_thread_num());
    // lancement du processus
  #pragma omp for schedule(static, N)
    for (i = 0; i < n_omp; i++) {
      d = 0.0; x = 0.0;
      while (1) {
        u = uniform_random_number();
        L = -(1 / c) * log(u);
        x = x + L * cos(d);
        if (x < 0) {
        	r++;
        	break;
        } else if (x >= h) {
        	t++;
        	break;
        } else if ((u = uniform_random_number()) < c_c / c) {
      	  b++;
          absorbed[j] = x;
        #pragma omp atomic
          j++;
        	break;
        } else {
        	u = uniform_random_number();
        	d = u * M_PI;
        }
      }
    }
  } /* FIN DE LA SECTION PARALLELE */
  
  // recuperation des neutrons absorbes calcules par le GPU
  for (i = 0; i < res_gpu[1]; i++)
    absorbed[j + i] = absorbed_gpu[i];
  
  // fin du chronometrage
  finish = my_gettimeofday();
  
  printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
  printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
  printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);
  
  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));
  
  // ouverture du fichier pour ecrire les positions des neutrons absorbés
  #if (OUTPUT == 1)
    FILE *f_handle = fopen(OUTPUT_FILE, "w");
    if (!f_handle) {
      fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
      exit(EXIT_FAILURE);
    }
    
    for (j = 0; j < b; j++)
      fprintf(f_handle, "%f\n", absorbed[j]);
    
    // fermeture du fichier
    fclose(f_handle);
    printf("Result written in " OUTPUT_FILE "\n");
  #endif
  
  // liberation memoire
  free(absorbed); free(res_gpu); free(absorbed_gpu);

  return EXIT_SUCCESS;
}


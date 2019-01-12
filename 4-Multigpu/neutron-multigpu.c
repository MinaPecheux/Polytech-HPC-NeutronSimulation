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

#define OUTPUT_FILE     "/tmp/absorbed.dat"

char info[] = "\
Usage:\n\
    neutron-multigpu H Nb C_c C_s nb_gpus\n\
\n\
    H  :     épaisseur de la plaque\n\
    Nb :     nombre d'échantillons\n\
    C_c:     composante absorbante\n\
    C_s:     componente diffusante\n\
    nb_gpus: nombre de GPUs actifs (<= 3)\n\
\n\
Exemple d'execution : \n\
    neutron-multigpu 1.0 500000000 0.5 0.5 1\n\
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
                             int** res, float** absorbed, int nb_gpus);

/*
 * main()
 */
int main(int argc, char *argv[]) {
  // La distance moyenne entre les interactions neutron/atome est 1/c. 
  // c_c et c_s sont les composantes absorbantes et diffusantes de c. 
  float c, c_c, c_s;
  // épaisseur de la plaque
  float h;
  // nombre d'échantillons
  int n;
  // nombre de neutrons refléchis, absorbés et transmis
  int r, b, t;
  // chronometrage
  double start, finish;
  
  int nb_gpus; // nombre de GPUs actifs

  if(argc == 1)
    fprintf( stderr, "%s\n", info);

  // valeurs par defaut
  h = 1.0;
  n = 500000000;
  c_c = 0.5;
  c_s = 0.5;
  nb_gpus = 1;

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
    nb_gpus = atof(argv[5]);
  r = b = t = 0;
  c = c_c + c_s;

  /* Verification du nombre de GPUs */
  if (nb_gpus > 3) {
      printf("Erreur : cet ordinateur ne dispose que de 3 GPUs !\n");
      exit(EXIT_FAILURE);
  }

  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);

  // contient : r, b, t, id du dernier neutron absorbe
  int   *res      = (int *)   calloc(4, sizeof(int));
  float *absorbed = (float *) calloc(n, sizeof(float));

  // debut du chronometrage
  start = my_gettimeofday();
  
  // simulation
  /* Probleme de driver ?
   * Il semble que la gestion memoire est bonne mais qu'il y a des erreurs avec
   * plusieurs GPUs. Le probleme pourrait venir d'une configuration CUDA
   * incompatible.
   * Cf : https://devtalk.nvidia.com/default/topic/470566/cuda-programming-and-performance/why-quot-all-cuda-capable-devices-are-busy-or-unavailable-quot-/
   */
  gpu_simulation(c, c_c, c_s, h, n, &res, &absorbed, nb_gpus);
  // recuperation des resultats (nb de neutrons reflechis, absorbes et transmis)
  r = res[0]; b = res[1]; t = res[2];
  
  // fin du chronometrage
  finish = my_gettimeofday();
  
  printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
  printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
  printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);
  
  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));
  
  // ouverture du fichier pour ecrire les positions des neutrons absorbés
  #if (OUTPUT == 1)
    int j = 0;
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
  free(res); free(absorbed);

  return EXIT_SUCCESS;
}


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
#ifdef _OPENMP
#include <omp.h>
#endif

#define N               2000
#define OUTPUT_FILE     "/tmp/absorbed.dat"

char info[] = "\
Usage:\n\
    neutron-openmp H Nb C_c C_s nb_threads\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
    nb_threads: nombre de thread OMP\n\
\n\
Exemple d'execution : \n\
    neutron-openmp 1.0 500000000 0.5 0.5 8\n\
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
  int n;
  // nombre de neutrons refléchis, absorbés et transmis
  int r, b, t;
  // chronometrage
  double start, finish;
  int i, j = 0; // compteurs
  // nombre de threads OMP
  int nb_threads;

  if( argc == 1)
    fprintf( stderr, "%s\n", info);

  // valeurs par defaut
  h = 1.0;
  n = 500000000;
  c_c = 0.5;
  c_s = 0.5;
  nb_threads = 8;

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
  r = b = t = 0;
  c = c_c + c_s;

  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);


  float *absorbed;
  absorbed = (float *) calloc(n, sizeof(float));

  // debut du chronometrage
  start = my_gettimeofday();
    
  // definition du nombre total de threads
  omp_set_num_threads(nb_threads);
  
  printf("--------\n");
  printf("[CPU] Nombre de threads OpenMP: %d.\n", nb_threads);

  /* DEBUT DE LA SECTION PARALLELE */
  #pragma omp parallel private(d, x, u, L, i) reduction(+:r,b,t)
  {
    // init du generateur avec le numero de thread
    init_uniform_random_number(omp_get_thread_num());
    // lancement du processus
  #pragma omp for schedule(static, N)
    for (i = 0; i < n; i++) {
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
  printf("r = %d, b = %d, t = %d\n", r, b, t);
  printf("Total treated: %d\n", r + b + t);

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
  free(absorbed);

  return EXIT_SUCCESS;
}


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
#include <mpi.h>

#define OUTPUT_FILE     "/tmp/absorbed.dat"

char info[] = "\
Usage:\n\
    neutron-hybrid_mpi H Nb C_c C_s gpu_cpu_ratio\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
    gpu_cpu_ratio: ratio d'equilibrage GPU/CPU (0.0: only GPU)\n\
\n\
Exemple d'execution : \n\
    neutron-hybrid_mpi 1.0 500000000 0.5 0.5 1.0\n\
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
  int n, n_cpu, n_gpu;
  // nombre de neutrons refléchis, absorbés et transmis
  int r, b, t;
  // chronometrage
  double start, finish;
  int i, j = 0; // compteurs
  
  double gpu_cpu_ratio;

  /* MPI: variables */
  int         rank, p;      // processors
  MPI_Status  status;

  if(argc == 1)
    fprintf(stderr, "%s\n", info);

  // valeurs par defaut
  h = 1.0;
  n = 500000000;
  c_c = 0.5;
  c_s = 0.5;
  gpu_cpu_ratio = 0.65;

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
    gpu_cpu_ratio = atof(argv[5]);
  r = b = t = 0;
  c = c_c + c_s;
  
  /* MPI: Initialisation */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // debut du chronometrage
  start = my_gettimeofday();

  // proc master gere le GPU et les transferts memoire
  if (rank == 0) {
    // affichage des parametres pour verificatrion
    printf("Épaisseur de la plaque : %4.g\n", h);
    printf("Nombre d'échantillons  : %d\n", n);
    printf("C_c : %g\n", c_c);
    printf("C_s : %g\n", c_s);
    
    // envoi du nombre de neutrons a traiter aux autres procs
    n_cpu = (int)(gpu_cpu_ratio * (float)n);
    n_gpu = n - n_cpu;
    printf("\nRatio d'equilibrage CPU / GPU : %g\n", gpu_cpu_ratio);
    printf("Nombre d'echantillons CPU : %d\n", n_cpu);
    printf("Nombre d'echantillons GPU : %d\n", n_gpu);
    
    int *nb_neutrons;
    int tmp = 0;
    if (n_gpu > 0) {
      nb_neutrons = (int*) calloc(p-1, sizeof(int));
      for (i = 0; i < p-1; i++) {
        nb_neutrons[i] = n_cpu / (p-1); // le proc 0 ne traite pas de neutrons
        tmp += nb_neutrons[i];
      }
      nb_neutrons[p-2] += n_cpu - tmp;
      printf("\n-------------\nNombre d'echantillons pour chaque proc MPI :\n");
      for (i = 0; i < p-1; i++)
        printf("Proc #%d: %d\n", i+1, nb_neutrons[i]);
    }
    else {
      printf("\nGPU deactivated !\n");
      nb_neutrons = (int*) calloc(p, sizeof(int));
      for (i = 0; i < p; i++) {
        nb_neutrons[i] = n_cpu / p;
        tmp += nb_neutrons[i];
      }
      nb_neutrons[p-1] += n_cpu - tmp;
      printf("\n-------------\nNombre d'echantillons pour chaque proc MPI :\n");
      for (i = 0; i < p; i++)
        printf("Proc #%d: %d\n", i, nb_neutrons[i]);
    }
      
    for (i = 1; i < p; i++)
      MPI_Send(nb_neutrons + (i - (n_gpu > 0)), 1, MPI_INT, i, 1, MPI_COMM_WORLD);

    // preparation memoire
    float *absorbed     = (float *) calloc(n, sizeof(float));
    // contient : r, b, t, id du dernier neutron absorbe
    int   *res          = (int *)   calloc(4, sizeof(int));
    float *absorbed_tmp = (float *) calloc(n_gpu, sizeof(float));
    
    // verification : traitement GPU desactive ?
    if (n_gpu > 0) {
      // simulation
      gpu_simulation(c, c_c, c_s, h, n_gpu, &res, &absorbed_tmp);
      // recuperation des resultats (nb de neutrons reflechis, absorbes et transmis)
      r += res[0]; b += res[1]; t += res[2];
  
      // recuperation des neutrons absorbes calcules par le GPU
      for (j = 0; j < res[1]; j++)
        absorbed[j] = absorbed_tmp[j];
    }
    // sinon : on switche sur un traitement CPU seul
    else {
      // initialisation du generateur pseudo-aleatoire
      init_uniform_random_number(rank);
      // debut du traitement
      for (i = 0; i < nb_neutrons[0]; i++) {
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
            absorbed[j++] = x;
          	break;
          } else {
          	u = uniform_random_number();
          	d = u * M_PI;
          }
        }
      }
    }
    
    // recuperation des resultats des autres procs
    for (i = 1; i < p; i++) {
      MPI_Recv(res, 4, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
      r += res[0]; b += res[1]; t += res[2];
      MPI_Recv(absorbed+j, res[1], MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
      j += res[1];
    }
    
    printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
    printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
    printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);

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
    free(nb_neutrons); free(absorbed); free(res); free(absorbed_tmp);
  }
  // les autres procs calculent sur CPU
  else {
    // recuperation du nombre de neutrons a traiter depuis le proc master
    MPI_Recv(&n, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
      
    // preparation memoire
    float *absorbed = (float *) calloc(n, sizeof(float));
    
    // initialisation du generateur pseudo-aleatoire
    init_uniform_random_number(rank);
    // debut du traitement
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
          absorbed[j++] = x;
        	break;
        } else {
        	u = uniform_random_number();
        	d = u * M_PI;
        }
      }
    }
    
    // renvoi des resultats au proc master
    int res[] = { r, b, t, 0 };
    MPI_Send(res, 4, MPI_INT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(absorbed, b, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    
    // liberation memoire
    free(absorbed);
  }

  /* MPI: Desactivation */
  MPI_Finalize();
  
  // fin du chronometrage
  finish = my_gettimeofday();
  printf("\n[MPI proc #%d] Temps total de calcul: %.8g sec\n",
    rank, finish - start);
  printf("[MPI proc #%d] Millions de neutrons /s: %.2g\n",
    rank, (double) n / ((finish - start)*1e6));

  return EXIT_SUCCESS;
}


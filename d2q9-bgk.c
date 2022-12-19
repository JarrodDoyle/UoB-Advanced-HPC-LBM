/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int nprocs;
  int rank;
  int neighbour_top;
  int neighbour_bot;
  int start_y;
  int end_y;
  int nrows;
  int nx;           /* no. of cells in x-direction */
  int ny;           /* no. of cells in y-direction */
  int maxIters;     /* no. of iterations */
  int reynolds_dim; /* dimension for Reynolds number */
  float density;    /* density per link */
  float accel;      /* density redistribution */
  float omega;      /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float *restrict __attribute__((aligned(64))) speed0;
  float *restrict __attribute__((aligned(64))) speed1;
  float *restrict __attribute__((aligned(64))) speed2;
  float *restrict __attribute__((aligned(64))) speed3;
  float *restrict __attribute__((aligned(64))) speed4;
  float *restrict __attribute__((aligned(64))) speed5;
  float *restrict __attribute__((aligned(64))) speed6;
  float *restrict __attribute__((aligned(64))) speed7;
  float *restrict __attribute__((aligned(64))) speed8;
} t_cells;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_cells *restrict final_cells, t_cells *restrict cells, t_cells *restrict tmp_cells,
               int **obstacles_ptr, int **fin_obstacles_ptr, float **av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_cells *restrict cells, t_cells *restrict tmp_cells, int *obstacles);
int accelerate_flow(const t_param params, t_cells *cells, int *obstacles);
int propagate(const t_param params, t_cells *restrict cells, t_cells *restrict tmp_cells);
float collision(const t_param params, t_cells *restrict cells, t_cells *restrict tmp_cells, int *obstacles);
int write_values(const t_param params, t_cells cells, int *obstacles, float *av_vels);
int halo_transfer(const t_param params, t_cells *restrict cells, int send_idx, int recv_idx, int dst, int src);

/* finalise, including freeing up allocated memory */
int finalise(const t_param *params, t_cells *restrict final_cells, t_cells *restrict cells, t_cells *restrict tmp_cells,
             int **final_obstacles_ptr, int **obstacles_ptr, float **av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_cells *cells);

/* compute average velocity */
float av_velocity(const t_param params, t_cells *cells, int *obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_cells *cells, int *obstacles);

/* utility functions */
void die(const char *message, const int line, const char *file);
void usage(const char *exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char *argv[])
{
  // Set-up MPI distribution environment
  MPI_Init(&argc, &argv);

  char *paramfile = NULL;                                                            /* name of the input parameter file */
  char *obstaclefile = NULL;                                                         /* name of a the input obstacle file */
  t_param params;                                                                    /* struct to hold parameter values */
  t_cells *final_cells = (t_cells *)_mm_malloc(sizeof(t_cells), 64);                 /* grid containing final fluid densities */
  t_cells *cells = (t_cells *)_mm_malloc(sizeof(t_cells), 64);                       /* grid containing fluid densities */
  t_cells *tmp_cells = (t_cells *)_mm_malloc(sizeof(t_cells), 64);                   /* scratch space */
  int *obstacles = NULL;                                                             /* grid indicating which cells are blocked */
  int *final_obstacles = NULL;
  float *vels = NULL;                                                                /* a record of the total velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  MPI_Comm_size(MPI_COMM_WORLD, &params.nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &params.rank);

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic = tot_tic;
  initialise(paramfile, obstaclefile, &params, final_cells, cells, tmp_cells, &obstacles, &final_obstacles, &vels);
  // int *final_obstacles = (int *)_mm_malloc(sizeof(int) * (params.ny * params.nx), 64);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic = init_toc;

  __assume(params.nx % 128 == 0);
  __assume(params.ny % 128 == 0);

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    vels[tt] = timestep(params, cells, tmp_cells, obstacles);
    t_cells *tmp = cells;
    cells = tmp_cells;
    tmp_cells = tmp;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("velocity: %.12E\n", vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic = comp_toc;

  // !Collate data from ranks here

  // Collect the grid from the other ranks
  const int num_cells = params.nx * params.nrows;
  int recv_counts[params.nprocs];
  int displs[params.nprocs];
  MPI_Gather(&num_cells, 1, MPI_INT, &recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (params.rank == 0)
  {
    displs[0] = 0;
    for (int i = 1; i < params.nprocs; i++)
      displs[i] = displs[i - 1] + recv_counts[i - 1];
  }

  MPI_Gatherv(cells->speed0 + params.nx, num_cells, MPI_FLOAT, final_cells->speed0, (const int *)&recv_counts, (const int *)&displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(cells->speed1 + params.nx, num_cells, MPI_FLOAT, final_cells->speed1, (const int *)&recv_counts, (const int *)&displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(cells->speed2 + params.nx, num_cells, MPI_FLOAT, final_cells->speed2, (const int *)&recv_counts, (const int *)&displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(cells->speed3 + params.nx, num_cells, MPI_FLOAT, final_cells->speed3, (const int *)&recv_counts, (const int *)&displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(cells->speed4 + params.nx, num_cells, MPI_FLOAT, final_cells->speed4, (const int *)&recv_counts, (const int *)&displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(cells->speed5 + params.nx, num_cells, MPI_FLOAT, final_cells->speed5, (const int *)&recv_counts, (const int *)&displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(cells->speed6 + params.nx, num_cells, MPI_FLOAT, final_cells->speed6, (const int *)&recv_counts, (const int *)&displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(cells->speed7 + params.nx, num_cells, MPI_FLOAT, final_cells->speed7, (const int *)&recv_counts, (const int *)&displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(cells->speed8 + params.nx, num_cells, MPI_FLOAT, final_cells->speed8, (const int *)&recv_counts, (const int *)&displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Gatherv(obstacles + params.nx, num_cells, MPI_INT, final_obstacles, (const int *)&recv_counts, (const int *)&displs, MPI_INT, 0, MPI_COMM_WORLD);

  // Collect av_velocities
  float av_vels[params.maxIters];
  MPI_Reduce(vels, &av_vels, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (params.rank == 0)
  {
    int tot_cells = params.nx * params.ny;
    for (int i = 0; i < params.nx * params.ny; i++)
      tot_cells -= final_obstacles[i];
    for (int i = 0; i < params.maxIters; i++)
      av_vels[i] /= tot_cells;
  }

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  if (params.rank == 0)
  {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, final_cells, final_obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc - tot_tic);
    write_values(params, *final_cells, final_obstacles, av_vels);
    finalise(&params, final_cells, cells, tmp_cells, &final_obstacles, &obstacles, &vels);
  }

  // Close down MPI
  MPI_Finalize();

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_cells *restrict cells, t_cells *restrict tmp_cells, int *obstacles)
{
  accelerate_flow(params, cells, obstacles);
  halo_transfer(params, cells, params.nrows * params.nx, 0, params.neighbour_bot, params.neighbour_top);
  halo_transfer(params, cells, params.nx, (params.nrows + 1) * params.nx, params.neighbour_top, params.neighbour_bot);
  propagate(params, cells, tmp_cells);
  return collision(params, cells, tmp_cells, obstacles);
}

int accelerate_flow(const t_param params, t_cells *cells, int *obstacles)
{
  // Early return if the 2nd grid row isn't in this rank
  const int mod_y = params.ny - 2;
  if (mod_y < params.start_y || params.end_y < mod_y)
    return EXIT_FAILURE;

  /* Tell the compiler these are aligned */
  __assume_aligned(cells, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  __assume_aligned(obstacles, 64);

  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  __assume(params.nx % 128 == 0);
  __assume(params.ny % 128 == 0);

  const int y = mod_y - params.start_y + 1;
  const int start_idx = params.nx * y;
  for (int i = start_idx; i < start_idx + params.nx; i++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[i] && (cells->speed3[i] - w1) > 0.f && (cells->speed6[i] - w2) > 0.f && (cells->speed7[i] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speed1[i] += w1;
      cells->speed5[i] += w2;
      cells->speed8[i] += w2;
      /* decrease 'west-side' densities */
      cells->speed3[i] -= w1;
      cells->speed6[i] -= w2;
      cells->speed7[i] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int halo_transfer(const t_param params, t_cells *restrict cells, int send_idx, int recv_idx, int dst, int src)
{
  MPI_Status status;
  MPI_Sendrecv(cells->speed0 + send_idx, params.nx, MPI_FLOAT, dst, params.rank,
               cells->speed0 + recv_idx, params.nx, MPI_FLOAT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  MPI_Sendrecv(cells->speed1 + send_idx, params.nx, MPI_FLOAT, dst, params.rank,
               cells->speed1 + recv_idx, params.nx, MPI_FLOAT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  MPI_Sendrecv(cells->speed2 + send_idx, params.nx, MPI_FLOAT, dst, params.rank,
               cells->speed2 + recv_idx, params.nx, MPI_FLOAT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  MPI_Sendrecv(cells->speed3 + send_idx, params.nx, MPI_FLOAT, dst, params.rank,
               cells->speed3 + recv_idx, params.nx, MPI_FLOAT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  MPI_Sendrecv(cells->speed4 + send_idx, params.nx, MPI_FLOAT, dst, params.rank,
               cells->speed4 + recv_idx, params.nx, MPI_FLOAT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  MPI_Sendrecv(cells->speed5 + send_idx, params.nx, MPI_FLOAT, dst, params.rank,
               cells->speed5 + recv_idx, params.nx, MPI_FLOAT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  MPI_Sendrecv(cells->speed6 + send_idx, params.nx, MPI_FLOAT, dst, params.rank,
               cells->speed6 + recv_idx, params.nx, MPI_FLOAT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  MPI_Sendrecv(cells->speed7 + send_idx, params.nx, MPI_FLOAT, dst, params.rank,
               cells->speed7 + recv_idx, params.nx, MPI_FLOAT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  MPI_Sendrecv(cells->speed8 + send_idx, params.nx, MPI_FLOAT, dst, params.rank,
               cells->speed8 + recv_idx, params.nx, MPI_FLOAT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_cells *restrict cells, t_cells *restrict tmp_cells)
{
  /* Tell the compiler these are aligned */
  __assume_aligned(cells, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  __assume_aligned(tmp_cells, 64);
  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);

  /* Central cells don't change at all so we can just swap pointers */
  float *tmp = cells->speed0;
  cells->speed0 = tmp_cells->speed0;
  tmp_cells->speed0 = tmp;

  /* loop over _all_ cells */
  __assume(params.nx % 128 == 0);
  __assume(params.ny % 128 == 0);

  /* We don't need to transfer halo regions to the scratch space.
  ** They are not read from again until this time next timestep, at which
  ** point they have been overridden by the halo transfer */

  // #pragma omp parallel for schedule(static)
#pragma vector aligned
  for (int jj = 1; jj <= params.nrows; jj++)
  {
#pragma vector aligned
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int i = ii + jj * params.nx;

      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int y_n = params.nx;
      const int y_s = -params.nx;
      const int x_w = (ii == 0 ? params.nx : 0) - 1;
      const int x_e = 1 - ((ii + 1 == params.nx) ? params.nx : 0);

      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells->speed1[i] = cells->speed1[i + x_w];       /* east */
      tmp_cells->speed2[i] = cells->speed2[i + y_s];       /* north */
      tmp_cells->speed3[i] = cells->speed3[i + x_e];       /* west */
      tmp_cells->speed4[i] = cells->speed4[i + y_n];       /* south */
      tmp_cells->speed5[i] = cells->speed5[i + x_w + y_s]; /* north-east */
      tmp_cells->speed6[i] = cells->speed6[i + x_e + y_s]; /* north-west */
      tmp_cells->speed7[i] = cells->speed7[i + x_e + y_n]; /* south-west */
      tmp_cells->speed8[i] = cells->speed8[i + x_w + y_n]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

float collision(const t_param params, t_cells *restrict cells, t_cells *restrict tmp_cells, int *obstacles)
{
  float tot_u = 0.0f; /* accumulated magnitudes of velocity for each cell */

  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* Tell the compiler these are aligned */
  __assume_aligned(cells, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  __assume_aligned(tmp_cells, 64);
  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);
  __assume_aligned(obstacles, 64);

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  __assume(params.nx % 128 == 0);
  __assume(params.ny % 128 == 0);
#pragma vector aligned
  // #pragma omp parallel for simd reduction(+                                 \
//                                         : tot_u, tot_cells) schedule(simd \
//                                                                      : static)
  // for (int i = 0; i < params.nx * params.ny; i++)
  // {
  for (int jj = 1; jj <= params.nrows; jj++)
  {
#pragma vector aligned
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int i = ii + jj * params.nx;
      if (obstacles[i])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        float tmp = tmp_cells->speed1[i];
        tmp_cells->speed1[i] = tmp_cells->speed3[i];
        tmp_cells->speed3[i] = tmp;
        tmp = tmp_cells->speed2[i];
        tmp_cells->speed2[i] = tmp_cells->speed4[i];
        tmp_cells->speed4[i] = tmp;
        tmp = tmp_cells->speed5[i];
        tmp_cells->speed5[i] = tmp_cells->speed7[i];
        tmp_cells->speed7[i] = tmp;
        tmp = tmp_cells->speed6[i];
        tmp_cells->speed6[i] = tmp_cells->speed8[i];
        tmp_cells->speed8[i] = tmp;
      }
      else
      {
        /* compute local density total */
        const float local_density =
            tmp_cells->speed0[i] + tmp_cells->speed1[i] + tmp_cells->speed2[i] +
            tmp_cells->speed3[i] + tmp_cells->speed4[i] + tmp_cells->speed5[i] +
            tmp_cells->speed6[i] + tmp_cells->speed7[i] + tmp_cells->speed8[i];

        /* compute velocity squared*/
        const float u_x = (tmp_cells->speed1[i] + tmp_cells->speed5[i] + tmp_cells->speed8[i] - (tmp_cells->speed3[i] + tmp_cells->speed6[i] + tmp_cells->speed7[i])) / local_density;
        const float u_y = (tmp_cells->speed2[i] + tmp_cells->speed5[i] + tmp_cells->speed6[i] - (tmp_cells->speed4[i] + tmp_cells->speed7[i] + tmp_cells->speed8[i])) / local_density;
        const float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] = u_x;        /* east */
        u[2] = u_y;        /* north */
        u[5] = u_x + u_y;  /* north-east */
        u[6] = -u_x + u_y; /* north-west */

        /* relaxation step */
        const float u_sqq = 1.f - (u_sq * 1.5f);
        const float w1d = w1 * local_density;
        const float w2d = w2 * local_density;
        tmp_cells->speed0[i] += params.omega * (w0 * local_density * u_sqq - tmp_cells->speed0[i]);
        u[0] = (u[1] * u[1]) * 4.5f + u_sqq;
        tmp_cells->speed1[i] += params.omega * (w1d * (3.f * u[1] + u[0]) - tmp_cells->speed1[i]);
        tmp_cells->speed3[i] += params.omega * (w1d * (-3.f * u[1] + u[0]) - tmp_cells->speed3[i]);
        u[0] = (u[2] * u[2]) * 4.5f + u_sqq;
        tmp_cells->speed2[i] += params.omega * (w1d * (3.f * u[2] + u[0]) - tmp_cells->speed2[i]);
        tmp_cells->speed4[i] += params.omega * (w1d * (-3.f * u[2] + u[0]) - tmp_cells->speed4[i]);
        u[0] = (u[5] * u[5]) * 4.5f + u_sqq;
        tmp_cells->speed5[i] += params.omega * (w2d * (3.f * u[5] + u[0]) - tmp_cells->speed5[i]);
        tmp_cells->speed7[i] += params.omega * (w2d * (-3.f * u[5] + u[0]) - tmp_cells->speed7[i]);
        u[0] = (u[6] * u[6]) * 4.5f + u_sqq;
        tmp_cells->speed6[i] += params.omega * (w2d * (3.f * u[6] + u[0]) - tmp_cells->speed6[i]);
        tmp_cells->speed8[i] += params.omega * (w2d * (-3.f * u[6] + u[0]) - tmp_cells->speed8[i]);

        /* accumulate the magnitude of the velocity */
        tot_u += sqrtf(u_sq);
      }
    }
  }

  return tot_u;
}

float av_velocity(const t_param params, t_cells *cells, int *obstacles)
{
  int tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.0f; /* accumulated magnitudes of velocity for each cell */

  /* Tell the compiler these are aligned */
  __assume_aligned(cells, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  __assume_aligned(obstacles, 64);

  /* loop over all non-blocked cells */
  __assume(params.nx % 128 == 0);
  __assume(params.ny % 128 == 0);
  for (int i = 0; i < params.nx * params.ny; i++)
  {
    /* ignore occupied cells */
    if (obstacles[i])
      continue;

    /* local density total */
    const float local_density =
        cells->speed0[i] + cells->speed1[i] + cells->speed2[i] +
        cells->speed3[i] + cells->speed4[i] + cells->speed5[i] +
        cells->speed6[i] + cells->speed7[i] + cells->speed8[i];

    /* accumulate the magnitude of the velocity */
    float u_x = (cells->speed1[i] + cells->speed5[i] + cells->speed8[i] - (cells->speed3[i] + cells->speed6[i] + cells->speed7[i])) / local_density;
    float u_y = (cells->speed2[i] + cells->speed5[i] + cells->speed6[i] - (cells->speed4[i] + cells->speed7[i] + cells->speed8[i])) / local_density;
    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
    ++tot_cells;
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_cells *restrict final_cells,
               t_cells *restrict cells, t_cells *restrict tmp_cells,
               int **obstacles_ptr, int **fin_obstacles_ptr, float **av_vels_ptr)
{
  char message[1024]; /* message buffer */
  FILE *fp;           /* file pointer */
  int xx, yy;         /* generic array indices */
  int blocked;        /* indicates whether a cell is blocked by an obstacle */
  int retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1)
    die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1)
    die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1)
    die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1)
    die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1)
    die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1)
    die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1)
    die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  // Work out above/below rank ids
  params->neighbour_top = (params->nprocs + params->rank - 1) % params->nprocs;
  params->neighbour_bot = (params->rank + 1) % params->nprocs;

  // Calculate start and end columns for the rank
  int base_height = params->ny / params->nprocs;
  int remainder = params->ny % params->nprocs;
  params->nrows = base_height + (params->rank < remainder ? 1 : 0);
  params->start_y = base_height * params->rank + (params->rank <= remainder ? params->rank : remainder);
  params->end_y = params->start_y + params->nrows - 1;

  if (params->rank == 0)
  {
    /* grid for collation on master rank */
    final_cells->speed0 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    final_cells->speed1 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    final_cells->speed2 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    final_cells->speed3 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    final_cells->speed4 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    final_cells->speed5 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    final_cells->speed6 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    final_cells->speed7 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
    final_cells->speed8 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

    if (
        final_cells->speed0 == NULL || final_cells->speed1 == NULL || final_cells->speed2 == NULL ||
        final_cells->speed2 == NULL || final_cells->speed4 == NULL || final_cells->speed5 == NULL ||
        final_cells->speed6 == NULL || final_cells->speed7 == NULL || final_cells->speed8 == NULL)
      die("cannot allocate memory for final_cells", __LINE__, __FILE__);

    /* the final map of obstacles */
    *fin_obstacles_ptr = _mm_malloc(sizeof(int) * (params->ny * params->nx), 64);

    if (*fin_obstacles_ptr == NULL)
      die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
  }

  /* main grid */
  cells->speed0 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  cells->speed1 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  cells->speed2 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  cells->speed3 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  cells->speed4 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  cells->speed5 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  cells->speed6 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  cells->speed7 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  cells->speed8 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);

  if (cells->speed0 == NULL || cells->speed1 == NULL || cells->speed2 == NULL ||
      cells->speed2 == NULL || cells->speed4 == NULL || cells->speed5 == NULL ||
      cells->speed6 == NULL || cells->speed7 == NULL || cells->speed8 == NULL)
    die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  tmp_cells->speed0 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  tmp_cells->speed1 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  tmp_cells->speed2 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  tmp_cells->speed3 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  tmp_cells->speed4 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  tmp_cells->speed5 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  tmp_cells->speed6 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  tmp_cells->speed7 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);
  tmp_cells->speed8 = (float *)_mm_malloc(sizeof(float) * ((params->nrows + 2) * params->nx), 64);

  if (tmp_cells->speed0 == NULL || tmp_cells->speed1 == NULL || tmp_cells->speed2 == NULL ||
      tmp_cells->speed2 == NULL || tmp_cells->speed4 == NULL || tmp_cells->speed5 == NULL ||
      tmp_cells->speed6 == NULL || tmp_cells->speed7 == NULL || tmp_cells->speed8 == NULL)
    die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * ((params->nrows + 2) * params->nx), 64);

  if (*obstacles_ptr == NULL)
    die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density / 9.f;
  float w2 = params->density / 36.f;

  /* Tell the compiler these are aligned */
  __assume_aligned(cells, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  __assume_aligned(tmp_cells, 64);
  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);

  __assume(params->nx % 128 == 0);
  __assume(params->ny % 128 == 0);
#pragma vector aligned
#pragma omp simd
  // #pragma omp parallel for schedule(static)
  for (int i = 0; i < params->nx * (params->nrows + 2); i++)
  {
    /* centre */
    cells->speed0[i] = w0;
    /* axis directions */
    cells->speed1[i] = w1;
    cells->speed2[i] = w1;
    cells->speed3[i] = w1;
    cells->speed4[i] = w1;
    /* diagonals */
    cells->speed5[i] = w2;
    cells->speed6[i] = w2;
    cells->speed7[i] = w2;
    cells->speed8[i] = w2;

    /* set all cells in obstacle array to zero */
    // TODO: Can probably slightly reduce obstacles array size (don't need halo regions?)
    (*obstacles_ptr)[i] = 0;
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  int above_y = (params->ny + params->start_y - 1) % params->ny;
  int below_y = (params->end_y + 1) % params->ny;
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3)
      die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1)
      die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1)
      die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1)
      die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    // Interior cells
    if (params->start_y <= yy && yy <= params->end_y)
      (*obstacles_ptr)[xx + (yy - params->start_y + 1) * params->nx] = blocked;

    // Halo region cells
    if (yy == above_y)
      (*obstacles_ptr)[xx] = blocked;
    if (yy == below_y)
      (*obstacles_ptr)[xx + (params->nrows + 1) * params->nx] = blocked;

    if (params->rank == 0)
      (*fin_obstacles_ptr)[xx + yy * params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float *)_mm_malloc(sizeof(float) * params->maxIters, 64);

  return EXIT_SUCCESS;
}

int finalise(const t_param *params, t_cells *restrict final_cells, t_cells *restrict cells, t_cells *restrict tmp_cells,
             int **final_obstacles_ptr, int **obstacles_ptr, float **av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  _mm_free(final_cells->speed0);
  _mm_free(final_cells->speed1);
  _mm_free(final_cells->speed2);
  _mm_free(final_cells->speed3);
  _mm_free(final_cells->speed4);
  _mm_free(final_cells->speed5);
  _mm_free(final_cells->speed6);
  _mm_free(final_cells->speed7);
  _mm_free(final_cells->speed8);

  _mm_free(cells->speed0);
  _mm_free(cells->speed1);
  _mm_free(cells->speed2);
  _mm_free(cells->speed3);
  _mm_free(cells->speed4);
  _mm_free(cells->speed5);
  _mm_free(cells->speed6);
  _mm_free(cells->speed7);
  _mm_free(cells->speed8);

  _mm_free(tmp_cells->speed0);
  _mm_free(tmp_cells->speed1);
  _mm_free(tmp_cells->speed2);
  _mm_free(tmp_cells->speed3);
  _mm_free(tmp_cells->speed4);
  _mm_free(tmp_cells->speed5);
  _mm_free(tmp_cells->speed6);
  _mm_free(tmp_cells->speed7);
  _mm_free(tmp_cells->speed8);

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*final_obstacles_ptr);
  *final_obstacles_ptr = NULL;

  _mm_free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_cells *cells, int *obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_cells *cells)
{
  /* Tell the compiler these are aligned */
  __assume_aligned(cells, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);

  float total = 0.f;
  for (int i = 0; i < params.nx * params.ny; i++)
  {
    total += cells->speed0[i] + cells->speed1[i] + cells->speed2[i] +
             cells->speed3[i] + cells->speed4[i] + cells->speed5[i] +
             cells->speed6[i] + cells->speed7[i] + cells->speed8[i];
  }

  return total;
}

int write_values(const t_param params, t_cells cells, int *obstacles, float *av_vels)
{
  FILE *fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float pressure;               /* fluid pressure in grid cell */
  float u_x;                    /* x-component of velocity in grid cell */
  float u_y;                    /* y-component of velocity in grid cell */
  float u;                      /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int i = jj * params.nx + ii;
      /* an occupied cell */
      if (obstacles[i])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        const float local_density =
            cells.speed0[i] + cells.speed1[i] + cells.speed2[i] +
            cells.speed3[i] + cells.speed4[i] + cells.speed5[i] +
            cells.speed6[i] + cells.speed7[i] + cells.speed8[i];

        /* compute x velocity component */
        u_x = (cells.speed1[i] + cells.speed5[i] + cells.speed8[i] - (cells.speed3[i] + cells.speed6[i] + cells.speed7[i])) / local_density;
        /* compute y velocity component */
        u_y = (cells.speed2[i] + cells.speed5[i] + cells.speed6[i] - (cells.speed4[i] + cells.speed7[i] + cells.speed8[i])) / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char *message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char *exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

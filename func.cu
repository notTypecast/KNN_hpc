#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* I/O routines */


/* Timer */
double gettime()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double) (tv.tv_sec+tv.tv_usec/1000000.0);
}

/* Function to approximate */
double fitfun(double *x, int n)
{
	double f = 0.0;
	int i;

#if 1
	for(i=0; i<n; i++)	/* circle */
		f += x[i]*x[i];
#endif
#if 0
	for(i=0; i<n-1; i++) {	/*  himmelblau */
		f = f + pow((x[i]*x[i]+x[i+1]-11.0),2) + pow((x[i]+x[i+1]*x[i+1]-7.0),2);
	}
#endif
#if 0
	for (i=0; i<n-1; i++)   /* rosenbrock */
		f = f + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);
#endif
#if 0
	for (i=0; i<n; i++)     /* rastrigin */
		f = f + pow(x[i],2) + 10.0 - 10.0*cos(2*M_PI*x[i]);
#endif

	return f;
}


/* random number generator  */
#define SEED_RAND()     srand48(1)
#define URAND()         drand48()

#ifndef LB
#define LB -1.0
#endif
#ifndef UB
#define UB 1.0
#endif

double get_rand(int k)
{
	return (UB-LB)*URAND()+LB;
}


/* utils */
double compute_min(double *v, int n)
{
	int i;
	double vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) vmin = v[i];

	return vmin;
}

double compute_max(double *v, int n)
{
	int i;
	double vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) vmax = v[i];

	return vmax;
}

double compute_sum(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s;
}

double compute_sum_pow(double *v, int n, int p)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i], p);

	return s;
}

double compute_mean(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s/n;
}

double compute_std(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return sqrt(s/(n-1));
}

double compute_var(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return s/n;
}

__device__ double compute_dist_square(double *v, double *w, int n, double *res) {
	*res = 0.0;
	for (int i = 0; i < n; ++i) {
		*res += pow(v[i] - w[i], 2);
	}
}

double compute_dist(double *v, double *w, int n)
{
	int i;
	double s = 0.0;
	for (i = 0; i < n; i++) {
		s+= pow(v[i]-w[i],2);
	}

	return sqrt(s);
}

double compute_max_pos(double *v, int n, int *pos)
{
	int i, p = 0;
	double vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) {
			vmax = v[i];
			p = i;
		}

	*pos = p;
	return vmax;
}

double compute_min_pos(double *v, int n, int *pos)
{
	int i, p = 0;
	double vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) {
			vmin = v[i];
			p = i;
		}

	*pos = p;
	return vmin;
}

double compute_root(double dist, int norm)
{
	if (dist == 0) return 0;

	switch (norm) {
	case 2:
		return sqrt(dist);
	case 1:
	case 0:
		return dist;
	default:
		return pow(dist, 1 / (double) norm);
	}
}

double compute_distance(double *pat1, double *pat2, int lpat, int norm)
{
	register int i;
	double dist = 0.0;

	for (i = 0; i < lpat; i++) {
		double diff = 0.0;

		diff = pat1[i] - pat2[i];

		switch (norm) {
		double   adiff;

		case 2:
			dist += diff * diff;
			break;
		case 1:
			dist += fabs(diff);
			break;
		case 0:
			if ((adiff = fabs(diff)) > dist)
			dist = adiff;
			break;
		default:
			dist += pow(fabs(diff), (double) norm);
			break;
		}
	}

	return dist;	// compute_root(dist);
}

__global__ void initialize_xdata(double **xdata_gpu, double *xmem_gpu, int elems) {
	int TOTAL_THREADS = gridDim.x*blockDim.x;
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int chunk = elems/TOTAL_THREADS;

	for (int i = tid*chunk; i < (tid+1)*chunk; ++i) xdata_gpu[i] = xmem_gpu + i*PROBDIM;
}

__device__ void insert_sorted_knn_list(double *nn_d, int *nn_x, int knn, double distance, int index) {
	// use binary search to find position in sorted knn list
	int left = 0;
    int right = knn-1;
	int new_index = -1;

	// find position of new element
	while (left <= right) {
		int middle = (left + right) / 2;
		if (distance <= nn_d[middle]) {
			right = middle - 1;
		}
		else {
			left = middle + 1;
		}
	}

	if (new_index == -1) {
		new_index = left;
	}

	// insert new element at calculated position
	for (int i = knn - 1; i > new_index; --i) {
		nn_d[i] = nn_d[i - 1];
		nn_x[i] = nn_x[i - 1];
	}
	nn_d[new_index] = distance;
	nn_x[new_index] = index;

}

__global__ void compute_knn_brute_force(double **xdata, double *q, int npat, int lpat, const int knn, int *nn_x, double *nn_d, double *sh_nn_d, int *sh_nn_x) {
	int TOTAL_THREADS = gridDim.x*blockDim.x;
	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	/* initialize pairs of index and distance */
	for (int i = knn - 1; i >= 0; --i) {
		sh_nn_x[tid*knn + i] = -1;
		sh_nn_d[tid*knn + i] = 1e99-i;
	}

	// last element of nn_d KNN list is neighbor with max current distance
	int THREAD_CHUNK = npat / TOTAL_THREADS;

	// loop training data
	double new_d;
	for (int i = tid*THREAD_CHUNK; i < (tid+1)*THREAD_CHUNK; ++i) {
		// euclidean, get squared distance to avoid sqrt(.)
		compute_dist_square(q, xdata[i], lpat, &new_d);
		if (i == 689278) {
			//printf("%d %lf\n", tid, new_d);
		}
		// compare distance to largest neighbor distance
		if (new_d < sh_nn_d[tid*knn + knn - 1]) {
			// add to sorted KNN list (using binary search)
			insert_sorted_knn_list(&sh_nn_d[tid*knn], &sh_nn_x[tid*knn], knn, new_d, i);
		}
	}

	/*if (tid == 21539) {
		for (int i = 0; i < knn; ++i) {
			printf("%d. dist: %lf, idx: %d\n", i+1, sh_nn_d[tid*knn + i], sh_nn_x[tid*knn + i]);
		}
	}*/
}

__global__ void merge_computed_arrays(double **xdata, double *q, int npat, int lpat, const int knn, int *nn_x, double *nn_d, double *sh_nn_d, int *sh_nn_x) {
	int TOTAL_THREADS = gridDim.x*blockDim.x;
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	double res_d[NNBS];
	int res_x[NNBS];

	if (tid == 0) {
		printf("Initial: %lf\n", sh_nn_d[21539*knn + 0]);
	}

	// merge knn arrays created by threads
	for (int step = 0; step < __log2f(TOTAL_THREADS); ++step) {
		// decide if this thread should run on this step
		if (!(tid % (int)(pow(2, step + 1)+0.5))) {
			int other_id = tid + (int)(pow(2, step)+0.5);
			int i = 0, j = 0;
			for (int curr = 0; curr < knn; ++curr) {
				if (sh_nn_d[tid*knn + i] < sh_nn_d[other_id*knn + j]) {
					res_d[curr] = sh_nn_d[tid*knn + i];
					res_x[curr] = sh_nn_x[tid*knn + i++];
				}
				else {
					res_d[curr] = sh_nn_d[other_id*knn + j];
					res_x[curr] = sh_nn_x[other_id*knn + j++];
				}
			}
			for (int curr = 0; curr < knn; ++curr) {
				//if (tid == 16384 && other_id==20480 && curr == 0) {
				if (fabs(res_d[curr] - 1.159182) < 1e-6 && fabs(res_d[curr] - sh_nn_d[tid*knn + curr]) > 1e-6) {
					printf("I am in %d which merged with %d in curr %d\n", tid, other_id, curr);
				}
				sh_nn_d[tid*knn + curr] = res_d[curr];
				sh_nn_x[tid*knn + curr] = res_x[curr];
			}
		}
	}
	if (tid == 0) {
		printf("Final: %lf\n", sh_nn_d[0]);
	}
	
	/*if (tid == 0) {
		for (int i = 0; i < knn; ++i) {
			printf("%d. dist: %lf, idx: %d\n", i+1, sh_nn_d[tid*knn + i], sh_nn_x[tid*knn + i]);
		}
	}*/
}


/* compute an approximation based on the values of the neighbors */
double predict_value(int dim, int knn, double *xdata, double *ydata, double *point, double *dist)
{
	int i;
	double sum_v = 0.0;
	// plain mean (other possible options: inverse distance weight, closest value inheritance)
	for (i = 0; i < knn; i++) {
		sum_v += ydata[i];
	}

	return sum_v/knn;
}


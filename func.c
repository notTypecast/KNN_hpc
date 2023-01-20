#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Timer */
float gettime()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (float) (tv.tv_sec+tv.tv_usec/1000000.0);
}

/* Function to approximate */
float fitfun(float *x, int n)
{
	float f = 0.f;
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

float get_rand(int k)
{
	return (float)(UB-LB)*URAND()+LB;
}


/* utils */
float compute_min(float *v, int n)
{
	int i;
	float vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) vmin = v[i];

	return vmin;
}

float compute_max(float *v, int n)
{
	int i;
	float vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) vmax = v[i];

	return vmax;
}

float compute_sum(float *v, int n)
{
	int i;
	float s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s;
}

float compute_sum_pow(float *v, int n, int p)
{
	int i;
	float s = 0;
	for (i = 0; i < n; i++) s += pow(v[i], p);

	return s;
}

float compute_mean(float *v, int n)
{
	int i;
	float s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s/n;
}

float compute_std(float *v, int n, float mean)
{
	int i;
	float s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return sqrt(s/(n-1));
}

float compute_var(float *v, int n, float mean)
{
	int i;
	float s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return s/n;
}

float compute_dist_square(float *v, float *w, int n) {
	float s = 0.0;
	for (int i = 0; i < n; ++i) {
		s += pow(v[i] - w[i], 2);
	}

	return s;
}

float compute_dist(float *v, float *w, int n)
{
	int i;
	float s = 0.0;
	for (i = 0; i < n; i++) {
		s+= pow(v[i]-w[i],2);
	}

	return sqrt(s);
}

float compute_max_pos(float *v, int n, int *pos)
{
	int i, p = 0;
	float vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) {
			vmax = v[i];
			p = i;
		}

	*pos = p;
	return vmax;
}

float compute_min_pos(float *v, int n, int *pos)
{
	int i, p = 0;
	float vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) {
			vmin = v[i];
			p = i;
		}

	*pos = p;
	return vmin;
}

float compute_root(float dist, int norm)
{
	if (dist == 0) return 0;

	switch (norm) {
	case 2:
		return sqrt(dist);
	case 1:
	case 0:
		return dist;
	default:
		return pow(dist, 1 / (float) norm);
	}
}

float compute_distance(float *pat1, float *pat2, int lpat, int norm)
{
	register int i;
	float dist = 0.0;

	for (i = 0; i < lpat; i++) {
		float diff = 0.0;

		diff = pat1[i] - pat2[i];

		switch (norm) {
		float   adiff;

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
			dist += pow(fabs(diff), (float) norm);
			break;
		}
	}

	return dist;	// compute_root(dist);
}


/* compute an approximation based on the values of the neighbors */
float predict_value(int dim, int knn, float *xdata, float *ydata, float *point, float *dist)
{
	int i;
	float sum_v = 0.0;
	// plain mean (other possible options: inverse distance weight, closest value inheritance)
	for (i = 0; i < knn; i++) {
		sum_v += ydata[i];
	}

	return sum_v/knn;
}


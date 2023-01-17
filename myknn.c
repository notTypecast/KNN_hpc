#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#ifndef PROBDIM
#define PROBDIM 2
#endif

#include "func.c"

static double **xdata;
static double ydata[TRAINELEMS];

#define MAX_NNB	256

double find_knn_value(double *p, int n, int knn)
{
	int nn_x[MAX_NNB];
	double nn_d[MAX_NNB];

	compute_knn_brute_force(xdata, p, TRAINELEMS, PROBDIM, knn, nn_x, nn_d); // brute-force /linear search

	double xd[MAX_NNB*PROBDIM];   // points
	double fd[MAX_NNB];     // function values

	for (int i = 0; i < knn; i++) {
		fd[i] = ydata[nn_x[i]];
	}

	for (int i = 0; i < knn; i++) {
		for (int j = 0; j < PROBDIM; j++) {
			xd[i*PROBDIM+j] = xdata[nn_x[i]][j];
		}
	}

	double fi;

	fi = predict_value(PROBDIM, knn, xd, fd, p, nn_d);

	return fi;
}

int main(int argc, char *argv[])
{
	// query data
	double *x = malloc(QUERYELEMS*PROBDIM*sizeof(double));
	double *y = malloc(QUERYELEMS*sizeof(double));

	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	double *xmem = (double *)malloc(TRAINELEMS*PROBDIM*sizeof(double));
	xdata = (double **)malloc(TRAINELEMS*sizeof(double *));
	for (int i = 0; i < TRAINELEMS; i++) xdata[i] = xmem + i*PROBDIM; //&mem[i*PROBDIM];

	FILE *fpin = fopen(trainfile, "rb");
	int b_rd = fread(xmem, sizeof(double), TRAINELEMS*PROBDIM, fpin);
	if (b_rd != TRAINELEMS*PROBDIM) {
		printf("error reading file %s\n", trainfile);
		exit(1);
	}
	#if defined(SURROGATES)
		b_rd = fread(ydata, sizeof(double), TRAINELEMS, fpin);
		if (b_rd != TRAINELEMS) {
			printf("error reading file %s\n", trainfile);
		exit(1);
	}
	#else
		memset(ydata, 0, TRAINELEMS*sizeof(double));
	#endif
	fclose(fpin);

	fpin = fopen(queryfile, "rb");
	b_rd = fread(x, sizeof(double), QUERYELEMS*PROBDIM, fpin);
	if (b_rd != QUERYELEMS*PROBDIM) {
		printf("error reading file %s\n", queryfile);
		exit(1);
	}
	#if defined(SURROGATES)
		b_rd = fread(y, sizeof(double), QUERYELEMS, fpin);
		if (b_rd != QUERYELEMS) {
			printf("error reading file %s\n", queryfile);
			exit(1);
		}
	#else
		memset(ydata, 0, QUERYELEMS*sizeof(double));
	#endif
	fclose(fpin);

	FILE *fpout = fopen("output.knn.txt","w");

	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;

	#pragma omp parallel for
	for (int i=0;i<QUERYELEMS;i++) {	/* requests */
		t0 = gettime();
		double yp = find_knn_value(&x[PROBDIM*i], PROBDIM, NNBS);
		t1 = gettime();
		t_sum += (t1-t0);
		if (i == 0) t_first = (t1-t0);

		sse += (y[i]-yp)*(y[i]-yp);

		for (int k = 0; k < PROBDIM; k++)
			fprintf(fpout,"%.5f ", x[k]);

		err = 100.0*fabs((yp-y[i])/y[i]);
		fprintf(fpout,"%.5f %.5f %.2f\n", y[i], yp, err);
		err_sum += err;

	}
	fclose(fpout);

	double mse = sse/QUERYELEMS;
	double ymean = compute_mean(y, QUERYELEMS);
	double var = compute_var(y, QUERYELEMS, ymean);
	double r2 = 1-(mse/var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum/QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	t_sum = t_sum*1000.0;			// convert to ms
	t_first = t_first*1000.0;	// convert to ms
	printf("Total time = %lf ms\n", t_sum);
	printf("Time for 1st query = %lf ms\n", t_first);
	printf("Time for 2..N queries = %lf ms\n", t_sum-t_first);
	printf("Average time/query = %lf ms\n", (t_sum-t_first)/(QUERYELEMS-1));

	return 0;
}

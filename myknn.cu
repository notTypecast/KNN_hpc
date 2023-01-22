#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); getchar(); }
    }
}

/* This code is very inefficient
 * See the improved implementation on cuda branch
 */

#ifndef PROBDIM
#define PROBDIM 2
#endif

#include "func.cu"

int TOTAL_THREADS = TRAINELEMS/NNBS;
int THREADS_PER_BLOCK = 128;

//static double **xdata;
static double ydata[TRAINELEMS];

static double **xdata_gpu;

#define MAX_NNB	256

double find_knn_value(double *p, int n, int knn, double *sh_nn_d, int *sh_nn_x)
{
	double *nn_d = (double *)malloc(knn*sizeof(double));
	int *nn_x = (int *)malloc(knn*sizeof(int));

	double *p_gpu;
	cudaMalloc(&p_gpu, n*sizeof(double));
	cudaMemcpy(p_gpu, p, n*sizeof(double), cudaMemcpyHostToDevice);

	compute_knn_brute_force<<<TOTAL_THREADS/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(xdata_gpu, p_gpu, TRAINELEMS, PROBDIM, knn, nn_x, nn_d, sh_nn_d, sh_nn_x); // brute-force /linear search
	cudaDeviceSynchronize();
	merge_computed_arrays<<<TOTAL_THREADS/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(xdata_gpu, p_gpu, TRAINELEMS, PROBDIM, knn, nn_x, nn_d, sh_nn_d, sh_nn_x);
	cudaDeviceSynchronize();
	
	gpuErrchk(cudaMemcpy(nn_d, sh_nn_d, knn*sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(nn_x, sh_nn_x, knn*sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(p_gpu);
	
	double xd[MAX_NNB*PROBDIM];   // points
	double fd[MAX_NNB];     // function values

	for (int i = 0; i < knn; i++) {
		fd[i] = ydata[nn_x[i]];
	}

	double fi;

	fi = predict_value(PROBDIM, knn, xd, fd, p, nn_d);

	free(nn_d);
	free(nn_x);

	return fi;
}

int main(int argc, char *argv[])
{
	if (argc != 3) {
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	// query data
	double *x = (double *)malloc(QUERYELEMS*PROBDIM*sizeof(double));
	double *y = (double *)malloc(QUERYELEMS*sizeof(double));

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	double *xmem = (double *)malloc(TRAINELEMS*PROBDIM*sizeof(double));
	// xdata = (double **)malloc(TRAINELEMS*sizeof(double *));
	// for (int i = 0; i < TRAINELEMS; i++) xdata[i] = xmem + i*PROBDIM; //&mem[i*PROBDIM];

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

	// allocate GPU memory
	double *x_gpu; 
	double *y_gpu;
	cudaMalloc(&x_gpu, QUERYELEMS*PROBDIM*sizeof(double));
	cudaMalloc(&y_gpu, QUERYELEMS*sizeof(double));

	double *xmem_gpu;
	cudaMalloc(&xmem_gpu, TRAINELEMS*PROBDIM*sizeof(double));
	cudaMalloc(&xdata_gpu, TRAINELEMS*sizeof(double));

	// declare shared memory for computations in GPU
	double *sh_nn_d;
	int *sh_nn_x;
	cudaMalloc(&sh_nn_d, TOTAL_THREADS*NNBS*sizeof(double));
	cudaMalloc(&sh_nn_x, TOTAL_THREADS*NNBS*sizeof(int));

	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;

	// copy data to GPU and time transfer
	t0 = gettime();
	cudaMemcpy(x_gpu, x, QUERYELEMS*PROBDIM*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, QUERYELEMS*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(xmem_gpu, xmem, TRAINELEMS*PROBDIM*sizeof(double), cudaMemcpyHostToDevice);
	initialize_xdata<<<TOTAL_THREADS/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(xdata_gpu, xmem_gpu, TRAINELEMS);
	t1 = gettime();

	t_sum += t1 - t0;

	FILE *fpout = fopen("output.knn.txt","w");
	for (int i=0;i<QUERYELEMS;i++) {	/* requests */
		t0 = gettime();
		double yp = find_knn_value(&x_gpu[PROBDIM*i], PROBDIM, NNBS, sh_nn_d, sh_nn_x);
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

	free(x);
	free(y);
	free(xmem);

	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(xmem_gpu);
	cudaFree(xdata_gpu);
	cudaFree(sh_nn_d);
	cudaFree(sh_nn_x);

	return 0;
}

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

#ifndef PROBDIM
#define PROBDIM 2
#endif

#include "func.c"

static float ydata[TRAINELEMS];

#define MAX_NNB	256

void computeSquaredDistanceMatrix(float *train, float *query, int T, int Q, int D, float *dist) {
	#pragma acc parallel loop deviceptr(train, query, dist)
	for (int i = 0; i < T*Q; ++i) {
		int train_offset = i/Q;
		int query_offset = i%Q;
		float sum = 0.0;
		#pragma acc loop seq independent
		for (int n = 0; n < D; ++n) {
			float tmp = train[n*T + train_offset] - query[n*Q + query_offset];
			sum += tmp*tmp;
		}

		dist[train_offset*Q + query_offset] = sum;
	}
}

void sortAndPredict(float *restrict dist, int *restrict idx, int T, int Q, int k, float *restrict train_eval, float *restrict predictions) {
	#pragma acc parallel loop deviceptr(dist, idx, train_eval, predictions) independent
	for (int qi = 0; qi < Q; ++qi) {
		// get column representing distances for query tx
		float *restrict query_dist = &dist[qi];
		int *restrict query_idx = &idx[qi];
		query_idx[0] = 0;

		// insertion sort
		#pragma acc loop seq independent
		for (int i = 1; i < T; ++i) {
			// get next distance in query column
			float curr_dist = query_dist[i*Q];
			int curr_idx = i;

			// if distance is larger than current largest, skip it
			// this is fine since we only want the first k distances to be sorted
			if (i >= k && curr_dist > query_dist[(k-1)*Q]) {
				continue;
			}

			// starting current index (if < k) or index k, move elements to the right
			// until the correct position for this element is found
			int j = i < k-1 ? i : k-1;
			#pragma acc loop seq independent
			while (j > 0 && query_dist[(j-1)*Q] > curr_dist) {
				query_dist[j*Q] = query_dist[(j-1)*Q];
				query_idx[j*Q] = query_idx[(j-1)*Q];
				--j;
			}

			// place element at the new position
			query_dist[j*Q] = curr_dist;
			query_idx[j*Q] = curr_idx;
		}

		// get evaluation sum for all k nearest neighbors and divide by k
		float prediction = 0.0f;
		#pragma acc loop seq independent
		for (int i = 0; i < k; ++i) {
			prediction += train_eval[query_idx[i*Q]];
		}
		prediction /= k;

		// write prediction
		predictions[qi] = prediction;
	}
}

int main(int argc, char *argv[])
{
	// query data
	float *x = (float *)malloc(QUERYELEMS*PROBDIM*sizeof(float));
	float *y = (float *)malloc(QUERYELEMS*sizeof(float));

	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	float *xmem = (float *)malloc(TRAINELEMS*PROBDIM*sizeof(float));

	FILE *fpin = fopen(trainfile, "rb");
	int b_rd = fread(xmem, sizeof(float), TRAINELEMS*PROBDIM, fpin);
	if (b_rd != TRAINELEMS*PROBDIM) {
		printf("error reading file %s\n", trainfile);
		exit(1);
	}
	#if defined(SURROGATES)
		b_rd = fread(ydata, sizeof(float), TRAINELEMS, fpin);
		if (b_rd != TRAINELEMS) {
			printf("error reading file %s\n", trainfile);
		exit(1);
	}
	#else
		memset(ydata, 0, TRAINELEMS*sizeof(float));
	#endif
	fclose(fpin);

	fpin = fopen(queryfile, "rb");
	b_rd = fread(x, sizeof(float), QUERYELEMS*PROBDIM, fpin);
	if (b_rd != QUERYELEMS*PROBDIM) {
		printf("error reading file %s\n", queryfile);
		exit(1);
	}
	#if defined(SURROGATES)
		b_rd = fread(y, sizeof(float), QUERYELEMS, fpin);
		if (b_rd != QUERYELEMS) {
			printf("error reading file %s\n", queryfile);
			exit(1);
		}
	#else
		memset(ydata, 0, QUERYELEMS*sizeof(float));
	#endif
	fclose(fpin);

	double t0, t_sum = 0.0;

	acc_set_device_num(0, acc_device_nvidia);

	// decide memory split
	size_t free = acc_get_property(0, acc_device_current, acc_property_free_memory);
	if (!free) {
		printf("Device unavailable!\n");
		exit(1);
	}
	int div = 1;
	// make sure there is enough memory for all allocations
	while (free < ((QUERYELEMS/div)*TRAINELEMS + 2*TRAINELEMS*NNBS + 3*(QUERYELEMS/div)*NNBS)*sizeof(float)) {
		while (QUERYELEMS % ++div);
	}

	int query_batch = QUERYELEMS/div;

	/* Allocate the following 2D arrays on the device:
	* DxT array for training data
	* DxQ array for query data
	* TxQ array for distances
	* kxQ array for training data indices 
	* where D: vector dimensions, T: training vectors, Q: query vectors and k: nearest neighbors
	*/
	t0 = gettime();
	float *train_dev = acc_malloc(TRAINELEMS*PROBDIM*sizeof(float));
	float *query_dev = acc_malloc(query_batch*PROBDIM*sizeof(float));
	float *dist_dev = acc_malloc(query_batch*TRAINELEMS*sizeof(float));
	int *idx_dev = acc_malloc(query_batch*NNBS*sizeof(float));

	// Allocate array for training data evaluations
	float *train_eval_dev = acc_malloc(TRAINELEMS*sizeof(float));

	// Allocate array for results (predictions)
	float *predictions_dev = acc_malloc(query_batch*sizeof(float));
	t_sum += gettime() - t0;

	// x_batch dimensions: PROBDIM x query_batch
	float *x_batch = (float *)malloc(query_batch*PROBDIM*sizeof(float));
	float *predictions = (float *)malloc(QUERYELEMS*sizeof(float));

	for (int i = 0; i < div; ++i) {
		for (int d = 0; d < PROBDIM; ++d) {
			memcpy(&x_batch[d*query_batch], &x[d*QUERYELEMS + i*query_batch], query_batch*sizeof(float));
		}
		// start timer
		t0 = gettime();

		// Transfer data to device
		acc_memcpy_to_device(train_dev, xmem, TRAINELEMS*PROBDIM*sizeof(float));
		acc_memcpy_to_device(query_dev, x_batch, query_batch*PROBDIM*sizeof(float));
		acc_memcpy_to_device(train_eval_dev, ydata, TRAINELEMS*sizeof(float));

		// Run acc kernel to compute distance matrix
		computeSquaredDistanceMatrix(train_dev, query_dev, TRAINELEMS, query_batch, PROBDIM, dist_dev);

		// Run acc kernel to sort first k distances
		sortAndPredict(dist_dev, idx_dev, TRAINELEMS, query_batch, NNBS, train_eval_dev, predictions_dev);
		
		acc_memcpy_from_device(&predictions[i*query_batch], predictions_dev, query_batch*sizeof(float));

		t_sum += gettime() - t0;
	}

	double sse = 0.0;
	double err_sum = 0.0;

	// calculate errors
	for (int i=0;i<QUERYELEMS;i++) {
		sse += (y[i]-predictions[i])*(y[i]-predictions[i]);
		err_sum += 100.0*fabs((predictions[i]-y[i])/y[i]);
	}

	double mse = sse/QUERYELEMS;
	double ymean = compute_mean(y, QUERYELEMS);
	double var = compute_var(y, QUERYELEMS, ymean);
	double r2 = 1-(mse/var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum/QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	t_sum = t_sum*1000.0;			// convert to ms
	printf("Total time = %lf ms\n", t_sum);
	printf("Average time/query = %lf ms\n", t_sum/QUERYELEMS);

	acc_free(train_dev);
	acc_free(query_dev);
    acc_free(dist_dev);
    acc_free(idx_dev);
	acc_free(train_eval_dev);

	return 0;
}

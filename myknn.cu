#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef PROBDIM
#define PROBDIM 2
#endif

#define BLOCK_DIM 16

#include "func.c"

static float ydata[TRAINELEMS];

#define MAX_NNB	256


/* CUDA Kernels */
__global__ void computeSquaredDistanceMatrix(float *train, size_t train_pitch, float *query, size_t query_pitch, int T, int Q, int D, float *dist) {
	/* Computes distance matrix of size TxQ
	 * dist[i][j] contains the distance between train vector i and query vector j
	 */
	int ty = BLOCK_DIM*blockIdx.y + threadIdx.y;
	int tx = BLOCK_DIM*blockIdx.x + threadIdx.x;

	if (ty < T && tx < Q) {
		float sum = 0.0f;
		for (int i = 0; i < D; ++i) {
			float val = train[i*train_pitch + ty] - query[i*query_pitch + tx];
			sum += val*val;
		}

		dist[ty*query_pitch + tx] = sum;
	}
}

__global__ void sortAndPredict(float *dist, int dist_pitch, int *idx, int idx_pitch, int T, int Q, int k, float *train_eval, float *predictions) {
	/* Sorts the elements of each column (=query) of the distance matrix, such that the first k values are sorted
	 * Calculates prediction for query based on those neighbors
	 */
	int tx = BLOCK_DIM*blockIdx.x + threadIdx.x;

	if (tx < Q) {
		// get column representing distances for query tx
		float *query_dist = &dist[tx];
		int *query_idx = &idx[tx];
		query_idx[0] = 0;

		// insertion sort
		for (int i = 1; i < T; ++i) {
			// get next distance in query column
			float curr_dist = query_dist[i*dist_pitch];
			int curr_idx = i;

			// if distance is larger than current largest, skip it
			// this is fine since we only want the first k distances to be sorted
			if (i >= k && curr_dist > query_dist[(k-1)*dist_pitch]) {
				continue;
			}

			// starting current index (if < k) or index k, move elements to the right
			// until the correct position for this element is found
			int j = i < k-1 ? i : k-1;
			while (j > 0 && query_dist[(j-1)*dist_pitch] > curr_dist) {
				query_dist[j*dist_pitch] = query_dist[(j-1)*dist_pitch];
				query_idx[j*idx_pitch] = query_idx[(j-1)*idx_pitch];
				--j;
			}

			// place element at the new position
			query_dist[j*dist_pitch] = curr_dist;
			query_idx[j*idx_pitch] = curr_idx;
		}


		// get evaluation sum for all k nearest neighbors and divide by k
		float prediction = 0.0f;
		for (int i = 0; i < k; ++i) {
			prediction += train_eval[query_idx[i*idx_pitch]];
		}
		prediction /= k;

		// write prediction
		predictions[tx] = prediction;		
	}
}

int main(int argc, char *argv[])
{
	// query data
	float *x = malloc(QUERYELEMS*PROBDIM*sizeof(float));
	float *y = malloc(QUERYELEMS*sizeof(float));

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

	float t0, t_sum;

	// start timer
	t0 = gettime();

	/* Allocate the following 2D arrays on the device:
	 * DxT array for training data
	 * DxQ array for query data
	 * TxQ array for distances
	 * kxQ array for training data indices 
	 * where D: vector dimensions, T: training vectors, Q: query vectors and k: nearest neighbors
	 */
	float *train_dev;
	float *query_dev;
	float *dist_dev;
	float *idx_dev;
	size_t train_pitch_bytes, query_pitch_bytes, dist_pitch_bytes, idx_pitch_bytes;

	int err0 = cudaMallocPitch((void **)&train_dev, &train_pitch_bytes, TRAINELEMS*sizeof(float), PROBDIM);
	int err1 = cudaMallocPitch((void **)&query_dev, &query_pitch_bytes, QUERYELEMS*sizeof(float), PROBDIM);
	int err2 = cudaMallocPitch((void **)&dist_dev, &dist_pitch_bytes, QUERYELEMS*sizeof(float), TRAINELEMS);
	int err3 = cudaMallocPitch((void **)&idx_dev, &idx_pitch_bytes, QUERYELEMS*sizeof(int), NNBS);

	// Allocate array for training data evaluations
	float *train_eval_dev;
	int err4 = cudaMalloc((void **)&train_eval_dev, TRAINELEMS*sizeof(float));

	// Allocate array for results (predictions)
	float *predictions_dev;
	int err5 = cudaMalloc((void **)&predictions_dev, QUERYELEMS*sizeof(float));

	if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess) {
        printf("cudaMallocPitch: failed to allocate memory\n");
		free(x);
		free(y);
		free(xmem);
		cudaFree(train_dev);
		cudaFree(query_dev);
		cudaFree(dist_dev);
		cudaFree(idx_dev); 
		cudaFree(train_eval_dev);
		cudaFree(predictions_dev);
        return 1;
    }
	
	size_t train_pitch = train_pitch_bytes / sizeof(float);
	size_t query_pitch = query_pitch_bytes / sizeof(float);
	size_t dist_pitch = dist_pitch_bytes / sizeof(float);
	size_t idx_pitch = idx_pitch_bytes / sizeof(int);

	// Transfer data to device
	err0 = cudaMemcpy2D(train_dev, train_pitch_bytes, xmem, TRAINELEMS*sizeof(float), TRAINELEMS*sizeof(float), PROBDIM, cudaMemcpyHostToDevice);
	err1 = cudaMemcpy2D(query_dev, query_pitch_bytes, x, QUERYELEMS*sizeof(float), QUERYELEMS*sizeof(float), PROBDIM, cudaMemcpyHostToDevice);
	err2 = cudaMemcpy(train_eval_dev, ydata, TRAINELEMS*sizeof(float), cudaMemcpyHostToDevice);
	if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("cudaMemcpy2D: failed to transfer data to device\n");
		free(x);
		free(y);
		free(xmem);
		cudaFree(train_dev);
		cudaFree(query_dev);
		cudaFree(dist_dev);
		cudaFree(idx_dev); 
		cudaFree(train_eval_dev);
		cudaFree(predictions_dev);
        return 1; 
    }

	/* Run CUDA kernell to compute distance matrix
	 * We have a total of Q*T threads
	 * Each thread computes the squared euclidian distance between one query and one training vector
	 */
	computeSquaredDistanceMatrix<<<dim3(QUERYELEMS/BLOCK_DIM, TRAINELEMS/BLOCK_DIM), dim3(BLOCK_DIM, BLOCK_DIM, 1)>>>(train_dev, train_pitch, query_dev, query_pitch, TRAINELEMS, QUERYELEMS, PROBDIM, dist_pitch);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) {
		printf("computeSquaredDistanceMatrix: error in kernel execution\n");
		free(x);
		free(y);
		free(xmem);
		cudaFree(train_dev);
		cudaFree(query_dev);
		cudaFree(dist_dev);
		cudaFree(index_dev);
		cudaFree(train_eval_dev);
		cudaFree(predictions_dev);
        return 1;
	}

	/* Run CUDA kernell to sort first k distances
	 * Each thread will perform the sorting operation for one query
	 */
	sortAndPredict<<<QUERYELEMS/(BLOCK_DIM*BLOCK_DIM), BLOCK_DIM*BLOCK_DIM>>>(dist_dev, dist_pitch, idx_dev, idx_pitch, TRAINELEMS, QUERYELEMS, NNBS, train_eval_dev, predictions_dev);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) {
		printf("sortAndPredict: error in kernel execution\n");
		free(x);
		free(y);
		free(xmem);
		cudaFree(train_dev);
		cudaFree(query_dev);
		cudaFree(dist_dev);
		cudaFree(index_dev);
		cudaFree(train_eval_dev);
		cudaFree(predictions_dev);
        return 1;
	}

	float *predictions = (float *)malloc(QUERYELEMS*sizeof(float));
	cudaMemcpy(predictions, predictions_dev, QUERYELEMS*sizeof(float), cudaMemcpyDeviceToHost);

	t_sum = gettime() - t0;

	float sse = 0.0;
	float err_sum = 0.0;

	// calculate errors
	for (int i=0;i<QUERYELEMS;i++) {
		sse += (y[i]-predictions[i])*(y[i]-predictions[i]);
		err_sum += 100.0*fabs((predictions[i]-y[i])/y[i]);
	}

	float mse = sse/QUERYELEMS;
	float ymean = compute_mean(y, QUERYELEMS);
	float var = compute_var(y, QUERYELEMS, ymean);
	float r2 = 1-(mse/var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum/QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	t_sum = t_sum*1000.0;			// convert to ms
	printf("Total time = %lf ms\n", t_sum);
	printf("Average time/query = %lf ms\n", t_sum/QUERYELEMS);

	free(x);
	free(y);
	free(xmem);

	cudaFree(train_dev);
	cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(index_dev);
	cudaFree(train_eval_dev);

	return 0;
}

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/* CUDA Kernels */
__global__ void computeSquaredDistanceMatrix(float *train, size_t train_pitch, float *query, size_t query_pitch, int T, int Q, int D, float *dist, size_t dist_pitch) {
	/* Computes distance matrix of size TxQ
	 * dist[i][j] contains the distance between train vector i and query vector j
	 */
	int ty = blockDim.y*blockIdx.y + threadIdx.y;
	int tx = blockDim.x*blockIdx.x + threadIdx.x;

	if (tx < T && ty < Q) {
		float sum = 0.0f;
		for (int i = 0; i < D; ++i) {
			float val = train[i*train_pitch + tx] - query[i*query_pitch + ty];
			sum += val*val;
		}

		dist[tx*dist_pitch + ty] = sum;
	}
}

__global__ void sortAndPredict(float *dist, int dist_pitch, int *idx, int idx_pitch, int T, int Q, int k, float *train_eval, float *predictions) {
	/* Sorts the elements of each column (=query) of the distance matrix, such that the first k values are sorted
	 * Calculates prediction for query based on those neighbors
	 */
	int tx = blockDim.x*blockIdx.x + threadIdx.x;

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
	float *x = (float *)malloc(QUERYELEMS*PROBDIM*sizeof(float));
	float *y = (float *)malloc(QUERYELEMS*sizeof(float));

	if (argc != 3 && argc != 4)
	{
		printf("usage: %s <trainfile> <queryfile> (<devicenum>)\n", argv[0]);
		exit(1);
	}

	char *trainfile = argv[1];
	char *queryfile = argv[2];
	int devnum = argc == 3 ? 0 : atoi(argv[3]);

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

	int err = cudaSetDevice(devnum);
	if (err != cudaSuccess) {
		printf("Couldn't set CUDA device\n");
		exit(1);
	}

	// decide memory split
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	int div = 1;
	// make sure there is enough memory for all allocations
	while (free < (((long)QUERYELEMS/div)*TRAINELEMS + 2*TRAINELEMS*NNBS + 3*(QUERYELEMS/div)*NNBS)*sizeof(float)) {
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
	float *train_dev;
	float *query_dev;
	float *dist_dev;
	int *idx_dev;
	size_t train_pitch_bytes, query_pitch_bytes, dist_pitch_bytes, idx_pitch_bytes;

	t0 = gettime();
	gpuErrchk(cudaMallocPitch((void **)&train_dev, &train_pitch_bytes, TRAINELEMS*sizeof(float), PROBDIM));
	gpuErrchk(cudaMallocPitch((void **)&query_dev, &query_pitch_bytes, query_batch*sizeof(float), PROBDIM));
	gpuErrchk(cudaMallocPitch((void **)&dist_dev, &dist_pitch_bytes, query_batch*sizeof(float), TRAINELEMS));
	gpuErrchk(cudaMallocPitch((void **)&idx_dev, &idx_pitch_bytes, query_batch*sizeof(int), NNBS));

	// Allocate array for training data evaluations
	float *train_eval_dev;
	gpuErrchk(cudaMalloc((void **)&train_eval_dev, TRAINELEMS*sizeof(float)));

	// Allocate array for results (predictions)
	float *predictions_dev;
	gpuErrchk(cudaMalloc((void **)&predictions_dev, query_batch*sizeof(float)));
	t_sum += gettime() - t0;
	
	size_t train_pitch = train_pitch_bytes / sizeof(float);
	size_t query_pitch = query_pitch_bytes / sizeof(float);
	size_t dist_pitch = dist_pitch_bytes / sizeof(float);
	size_t idx_pitch = idx_pitch_bytes / sizeof(int);

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
		gpuErrchk(cudaMemcpy2D(train_dev, train_pitch_bytes, xmem, TRAINELEMS*sizeof(float), TRAINELEMS*sizeof(float), PROBDIM, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy2D(query_dev, query_pitch_bytes, x_batch, query_batch*sizeof(float), query_batch*sizeof(float), PROBDIM, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(train_eval_dev, ydata, TRAINELEMS*sizeof(float), cudaMemcpyHostToDevice));

		/* Run CUDA kernel to compute distance matrix
		* We have a total of Q*T threads
		* Each thread computes the squared euclidian distance between one query and one training vector
		*/
		computeSquaredDistanceMatrix<<<dim3(TRAINELEMS/BLOCK_DIM, query_batch/BLOCK_DIM, 1), dim3(BLOCK_DIM, BLOCK_DIM, 1)>>>(train_dev, train_pitch, query_dev, query_pitch, TRAINELEMS, query_batch, PROBDIM, dist_dev, dist_pitch);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		/* Run CUDA kernel to sort first k distances
		* Each thread will perform the sorting operation for one query
		*/
		float xdim = query_batch/(BLOCK_DIM*BLOCK_DIM);
		sortAndPredict<<<xdim > 0 ? xdim : 1, BLOCK_DIM*BLOCK_DIM>>>(dist_dev, dist_pitch, idx_dev, idx_pitch, TRAINELEMS, query_batch, NNBS, train_eval_dev, predictions_dev);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		
		gpuErrchk(cudaMemcpy(&predictions[i*query_batch], predictions_dev, query_batch*sizeof(float), cudaMemcpyDeviceToHost));

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

	cudaFree(train_dev);
	cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(idx_dev);
	cudaFree(train_eval_dev);

	return 0;
}

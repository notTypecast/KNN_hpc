#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef PROBDIM
#define PROBDIM 2
#endif

#include "func.c"

int main(int argc, char *argv[])
{
	//float x[PROBDIM], y;
	// store all data in a single buffer
	// all y values are stored after all x values
	float *to_write = (float *)malloc(TRAINELEMS*(PROBDIM + 1)*sizeof(float));
	
	FILE *fp;

	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	SEED_RAND();	/* the training set is fixed */

	// create all data to write and save in to_write buffer
	// y values will be stored after all x values
	int y_start_idx = TRAINELEMS*PROBDIM;
	for (int i=0;i<TRAINELEMS;i++)
	{
		int row_idx = PROBDIM*i;
		for (int k = 0; k < PROBDIM; k++)
			to_write[row_idx + k] = get_rand(k);

		to_write[y_start_idx + i] = fitfun(&to_write[row_idx], PROBDIM);
	}

	// write training data to file
	fp = fopen(trainfile, "wb");
	int n = fwrite(to_write, sizeof(float), (PROBDIM + 1)*TRAINELEMS, fp);
	if (n != (PROBDIM + 1)*TRAINELEMS) {
		printf("Error writing training data to file.\n");
	}
	fclose(fp);

	printf("%d data points written to %s!\n", TRAINELEMS, trainfile);

	free(to_write);
	to_write = (float *)malloc(QUERYELEMS*(PROBDIM + 1)*sizeof(float));

	y_start_idx = QUERYELEMS*PROBDIM;
	for (int i=0;i<QUERYELEMS;i++)
	{
		int row_idx = PROBDIM*i;
		for (int k = 0; k < PROBDIM; k++)
			to_write[row_idx + k] = get_rand(k);

		to_write[y_start_idx + i] = fitfun(&to_write[row_idx], PROBDIM);
	}

	// write training data to file
	fp = fopen(queryfile, "wb");
	n = fwrite(to_write, sizeof(float), (PROBDIM + 1)*QUERYELEMS, fp);
	if (n != (PROBDIM + 1)*QUERYELEMS) {
		printf("Error writing query data to file.\n");
	}
	fclose(fp);

	printf("%d data points written to %s!\n", QUERYELEMS, queryfile);

	free(to_write);

	return 0;
}

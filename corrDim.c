/* corrDim.c */
/* Вычисление корреляционной размерности */
/* Тулебаев С.Д. */
#include <stdio.h>
#include <math.h>
#include <mem.h>
#include <stdlib.h>

void fitLeastSquareLine(int pointsNum, double X[], double Y[],
                        double *slopePtr, double *interceptPtr) {
#define ZERO 1.0E-30
  int index;
  double Xsum=0.0, Ysum=0.0, XYsum=0.0, XXsum=0.0, Xmean=0.0, Ymean=0.0,
         Xtemp, Ytemp;

  for (index = 0; index < pointsNum; index++) {
    Xtemp = X[index];  Ytemp = Y[index];
    Xsum += Xtemp; Ysum += Ytemp;
    XYsum += Xtemp*Ytemp;
    XXsum += Xtemp*Xtemp;
  }
  Xmean = Xsum/pointsNum;  Ymean = Ysum/pointsNum;
  Xtemp = XXsum - Xsum*Xmean;
  if (fabs(Xtemp) < ZERO)
    Xtemp = ((Xtemp < 0) ? -1 : +1)*ZERO;
  *slopePtr = (XYsum - Xsum*Ymean)/Xtemp;
  *interceptPtr = Ymean - *slopePtr*Xmean;
}

double calcSqrDistance(double *data_org, int first, int second, int dim) {
  int i;
  double *ptr1, *ptr2;
  double sum=0.0, temp;

  ptr1 = data_org + first*dim;
  ptr2 = data_org + second*dim;
  for (i=0; i<dim; i++, ptr1++, ptr2++) {
    temp = *ptr2 - *ptr1;
    sum += temp*temp;
  }
  return sum;
}

int main(int argc, char *argv[]) {
#define EXP_NUM 2046
#define BLOCK_NUM 1024
  FILE *fp;
  double *data;
  int dim, min_embed_dim=1, max_embed_dim=8, numFitPoints;
  unsigned int numData, numPoints, i, j;
  unsigned int Nd[EXP_NUM];
  unsigned short int *IEEE, exponent, min_exp, max_exp;
  double sum, corrDim, temp;
  double *logr, *logC;
  unsigned long NN1;
  const double log2 = log(2.0);

  if (argc < 2) {
    printf("Usage:  corrDim data_file [-d min_dim max_dim]\n");
    return -1;
  }
  fp = fopen(argv[1], "r");
  if (fp == NULL) {
    printf("Can't open datafile %s\n", argv[1]);
    return -1;
  }
  if ((argc>3) && (argv[2][0]=='-') && (argv[2][1]=='d')) {
    dim = atoi(argv[3]);
    if (dim != 0) min_embed_dim = dim;
    dim = atoi(argv[4]);
    if (dim != 0) max_embed_dim = dim;
  }
  data = NULL;
  numData = 0;
  i = 0;
  while (1) {
    if (i == numData) {
      numData += BLOCK_NUM;
      data = realloc(data, numData*sizeof(double));
      if (data == NULL) {
        printf("Can't allocate data in memory\n");
        free(data);
        return -1;
      }
    }
    if (fscanf(fp, "%lf", &data[i]) == EOF) break;
    i++;
  }
  numData = i;
  fclose(fp);
  for (dim = min_embed_dim; dim <= max_embed_dim; dim++) {
    numPoints = numData/dim;
    memset(Nd, 0, sizeof(Nd));
    for (i = 0; i < numPoints-1; i++)
      for (j = i+1; j < numPoints; j++) {
        temp = calcSqrDistance(data, i, j, dim);
        IEEE = (unsigned short int*)&temp; /* sizeof(int)==2 must be! */
        exponent = (IEEE[3] & 32767) >> 4;
        Nd[exponent]++;
      }
    min_exp = 1;
    while (Nd[min_exp]==0) min_exp++;
    max_exp = EXP_NUM-1;
    while (Nd[max_exp]==0) max_exp--;
    numFitPoints = max_exp-min_exp+1;
    logr = (double*) malloc(numFitPoints*sizeof(double));
    logC = (double*) malloc(numFitPoints*sizeof(double));
    NN1 = numPoints*(numPoints-1);
    sum = 0.0;
    for (i = 0, exponent = min_exp; exponent <= max_exp; exponent++, i++) {
      sum = sum + Nd[exponent];
      logr[i] = exponent/2.0;
      logC[i] = log(sum/NN1)/log2;
    }
    fitLeastSquareLine(numFitPoints, logr, logC, &corrDim, &temp);
    printf("Embedded dimension = %i\n", dim);
    printf("\tCorrelation dimension = %.5f\n", corrDim);
    free(logr); free(logC);
  }
  free(data);
  return 0;
}

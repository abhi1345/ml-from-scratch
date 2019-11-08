#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linreg.h"

/* For testing regression */
float rmse(float *y, float *yhat, size_t size) {
  float error = 0;
  for (int i=0; i < size; i++) {
    error += pow(yhat[i] - y[i], 2);
  }
  return pow(error / size, 0.5);
}

/* For testing classification */
float accuracy(float *y, float *yhat, size_t size) {
  float correct = 0;
  for (int i=0; i < size; i++) {
    if (yhat[i] == y[i]) {
      correct++;
    }
  }
  return correct / size;
}

int main(void) {
  printf("Linear Regression Test \n");
  size_t size = 7;
  float x[7] = {1,2,3,4,5,6,7}; 
  float y[7] = {3,5,7,9,11,13,15};
  struct LinearRegression *model = malloc(sizeof(struct LinearRegression));
  model_init(model);
  fit(model, x, y, size);
  float output[size];
  predict(model, x, size, output);
  printf("%lf is RMSE\n", rmse(output, y, size));
  return 0;
}

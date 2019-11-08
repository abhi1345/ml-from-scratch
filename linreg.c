#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "linreg.h"

void model_init(struct LinearRegression *model) {
  model->w = DEFAULT_W;
  model->a = DEFAULT_A;
}

void fit(struct LinearRegression *model, float *x, float *y, size_t size) {
  sgd(model, 1000, 0.08, x, y, size);
  return;
}

void sgd(struct LinearRegression *model, int iterations, float lr, float *x, float*y, size_t size) {
  for (int i=0; i < iterations; i++) {
    for (int i=0; i < size; i++) {
      float e = compute(model, x[i]) - y[i];
      /* Update constant*/
      model->a -= (lr * e);
      /* Update weight*/
      model->w -= (lr * e * x[i]);
    }
  }
  return;
}

float compute(struct LinearRegression *model, float x) {
  return model->w*x + model->a;
}

float mse(struct LinearRegression *model, float *x, float *y, size_t size) {
  float yhat[size];
  predict(model, x, size, yhat);
  float error = 0;
  for (int i=0; i < size; i++) {
    error += pow(yhat[i] - y[i], 2);
  }
  return error / size;
}

float *predict(struct LinearRegression *model, float *x, size_t size, float *output) {
  for (int i=0; i < size; i++) {
    output[i] = compute(model, x[i]);
  }
  return output;
}

/* Binary Logistic Regression with Stochastic Gradient Descent */
struct LogisticRegression {
  float w;
  float a;
};

#define DEFAULT_W 2.0;
#define DEFAULT_A 1.0;

/* Returns 0 if successful, -1 if failure */
void model_init(struct LinearRegression *model);

/* Learns slope and constant for linear model */
void fit(struct LinearRegression *model, float *x1, float *x2, int *y, size_t size);

/* Optimize weights using SGD using training data */
void sgd(struct LinearRegression *model, int iterations, float lr, float *x1, float *x2, int *y, size_t size);

/* Predict yhat for a single x value */
float compute(struct LinearRegression *model, float x1, float x2);

/* Logistic function */
float logit(struct LinearRegression *model, float x1, float x2);

/* Step function for final output*/
int step(float x);

/* Compute Mean Squared Error for a set of x,y */
float mse(struct LinearRegression *model, float *x1, float *x2, int *y, size_t size);

/* Return predictions for a set of inputs */
float *predict(struct LinearRegression *model, float *x1, float *x2, size_t size, float *output);

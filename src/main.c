#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CAT_LABEL 1.0
#define DOG_LABEL 0.0
#define EPOCHS 100
#define LEARNING_RATE 0.5

#define IMAGE_WIDTH 128
#define IMAGE_HEIGHT 128
#define IMAGE_CHANNELS 3
#define HIDDEN_SIZE 16
#define TRAIN_COUNT 100

float cat_train[TRAIN_COUNT][IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS] = {0};
float dog_train[TRAIN_COUNT][IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS] = {0};
float cat_test[TRAIN_COUNT][IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS] = {0};
float dog_test[TRAIN_COUNT][IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS] = {0};

//                            weights          inputs   size
float dot_product(const float *xs, const float *ys, int n) {
  float sum = 0;
  for (int i=0; i<n; i++) {
    sum += xs[i] * ys[i];
  }
  return sum;
}

void matrix_mul(const float *matrix, const float *xs, float *result, int rows, int cols){
  for (int i=0; i<rows; i++){
    result[i] = dot_product(matrix + i * cols, xs, cols);
  }
}

float rand11(){
  return (rand() / (float)RAND_MAX -0.5f) * 2.0;
}

void add_vec(const float *xs, const float *ys, float *result, int n){
  for (int i=0; i<n; i++){
    result[i] = xs[i] + ys[i];
  }
}

float sigmoid(float x){
  return 1.0f/ (1.0 + expf(-x));
}

void load_image(const char *filename, float *image){

  int width, height, channels;
  float *data = stbi_loadf(filename, &width, &height, &channels, IMAGE_CHANNELS);
  stbir_resize_float_linear(data, width, height, 0, image, IMAGE_WIDTH, IMAGE_HEIGHT, 0, STBIR_RGB);

  stbi_image_free(data);
}

void load_dataset(const char *path, const char *label, int offset, int count, float train[][IMAGE_WIDTH*IMAGE_WIDTH*IMAGE_CHANNELS]){
  for (int i=offset; i<offset+count; i++){
    char filename[128] = {0};
    sprintf(filename, "%s/%s.%d.jpg", path, label, i);
    load_image(filename, train[i]);
  }
}

typedef struct{
  float a0[IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS];
  float w1[IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS][HIDDEN_SIZE];
  float a1[HIDDEN_SIZE];
  float b1[HIDDEN_SIZE];
  float z1[HIDDEN_SIZE];
  float w2[HIDDEN_SIZE];
  float a2;
  float b2;
  float z2;
} neural_network;

void nn_init(neural_network *nn){
  for (int i=0; i<HIDDEN_SIZE; i++){
    for(int j=0; j<IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS; j++){
      nn->w1[j][i] = rand11();
    }
    nn->b1[i] = rand11();
  }  
  for (int i=0; i<HIDDEN_SIZE; i++){
    nn->w2[i] = rand11();
  }
  nn->b2 = rand11();
}

float feed_forward(neural_network *nn, float *x){
  memcpy(nn->a0, x, IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS*sizeof(float));
  matrix_mul((float *)nn->w1, x, nn->z1, HIDDEN_SIZE, IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS);
  add_vec(nn->z1, nn->b1, nn->z1, HIDDEN_SIZE);
  for (int i=0; i<HIDDEN_SIZE; i++){
    nn->a1[i] = sigmoid(nn->z1[i]);
  }

  nn->z2 = dot_product(nn->a1, nn->w2, HIDDEN_SIZE);
  nn->a2 = sigmoid(nn->z2);

  return nn->a2;
}

#define nn_loss(y, y_hat) (y-y_hat)*(y-y_hat);
// float nn_loss(float y, float y_hat){
//   return (y-y_hat) * (y-y_hat);
// }

void nn_gradient(neural_network *nn, float y_hat, neural_network *grad){
  float dC_da2 = 2* (nn->a2 - y_hat);
  float da2_dz2 = sigmoid(nn->a2) * (1-sigmoid(nn->a2));

  // want dC/dw1 = dC/da1 * dz1/da1 * dz1/dw1
  // want dC/da1 = dC/smth * smth/da1 = dC/da2 * da2/dz2 * dz2/da1
  for (int i=0; i<HIDDEN_SIZE; i++){
    float dC_da1 = dC_da2 * da2_dz2 * nn->w2[i];
    float da1_dz1 = sigmoid(nn->a1[i]) * (1-sigmoid(nn->a1[i]));
    for (int j=0; j<IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS; j++){
      float dz1_dw1 = nn->a0[j];
      grad->w1[j][i] = dC_da1*da1_dz1*dz1_dw1;
    }
    float dz1_db1 = 1;
    grad->b2 = dC_da1 * da1_dz1 * dz1_db1;
  }
  

  // want dC/dw2 = dC/da2 * dz2/da2 * dz2/dw2
  for (int i=0; i<HIDDEN_SIZE; i++){
    float dz2_dw2 = nn->a1[i];
    grad->w2[i] = dC_da2 * da2_dz2 * dz2_dw2;
  }

  float dz2_db2 = 1;
  grad->b2 = dC_da2 * da2_dz2 * dz2_db2;
}

void nn_backward(neural_network *nn, neural_network *grad, float learning_rate){
  for (int i=0; i<HIDDEN_SIZE; i++){
    for (int j=0; j<IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS; j++){
      nn->w1[j][i] -= grad->w1[j][i] * learning_rate;
    }
    nn->b1[i] -= grad->b1[i] * learning_rate;
  }

  for (int i=0; i<HIDDEN_SIZE; i++){
    nn->w2[i] -= grad->w2[i] * learning_rate;
  }
  nn->b2 -= grad->b2 * learning_rate;
}

void learn(neural_network *nn, float train[][IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS], float y_hat, float learning_rate){
  for (int i=0; i<TRAIN_COUNT; i++){
    neural_network grad_i;
    float y = feed_forward(nn, train[i]);
    nn_gradient(nn, y_hat, &grad_i);
    nn_backward(nn, &grad_i, learning_rate);
  }
}

float compute_loss(neural_network *nn, float train[][IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS], float y_hat){
  float loss = 0;
  for (int i=0; i<TRAIN_COUNT; i++){
    float y = feed_forward(nn, train[i]);
    loss += nn_loss(y, y_hat);
  }

  return loss;
}

int compute_true_positive(neural_network *nn, float train[][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS], float y_hat) {
  int count = 0;
  for (int i = 0; i < TRAIN_COUNT; i++) {
    float y = feed_forward(nn, train[i]);
    if (round(y) == y_hat) {
      count++;
    }
  }

  return count;
}

int main(int argc, char *argv[]){

  srand(time(NULL));
  int total = 0;
  int correct = 0;

  neural_network nn;
  nn_init(&nn);

  load_dataset("dvc/train/train", "cat", 0, TRAIN_COUNT, cat_train);
  load_dataset("dvc/train/train", "dog", 0, TRAIN_COUNT, dog_train);
  printf("TRAIN LOADED\n");
  load_dataset("dvc/train/train", "cat", TRAIN_COUNT, TRAIN_COUNT, cat_test);
  load_dataset("dvc/train/train", "dog", 0, TRAIN_COUNT, dog_test);
  printf("TEST LOADED\n");

  total = 2*TRAIN_COUNT;
  correct = compute_true_positive(&nn, cat_test, CAT_LABEL) + compute_true_positive(&nn, dog_test, DOG_LABEL);
  printf("%f\n", correct/(float)total);

  for (int i=0; i<EPOCHS; i++){
    learn(&nn, cat_train, CAT_LABEL, LEARNING_RATE);
    learn(&nn, dog_train, DOG_LABEL, LEARNING_RATE);

    float cat_loss = compute_loss(&nn, cat_train, CAT_LABEL);
    float dog_loss = compute_loss(&nn, dog_train, DOG_LABEL);
    float loss = (cat_loss+dog_loss)/2.0;
    if (loss < 0.05) break;
    printf("%f\n", loss);
  }

  total = 2*TRAIN_COUNT;
  correct = compute_true_positive(&nn, cat_test, CAT_LABEL) + compute_true_positive(&nn, dog_test, DOG_LABEL);
  printf("finale: %f\n", correct/(float)total);

  return 0;
}

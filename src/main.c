#include <signal.h>
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
#define LEARNING_RATE 0.01

#define IMAGE_WIDTH 128
#define IMAGE_HEIGHT 128
#define IMAGE_CHANNELS 3
#define TRAIN_COUNT 100

float cat_train[TRAIN_COUNT][IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS] = {0};
float dog_train[TRAIN_COUNT][IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS] = {0};

//                            weights          inputs   size
float dot_product(const float *xs, const float *ys, int n) {
  float sum = 0;
  for (int i=0; i<n; i++) {
    sum += xs[i] * ys[i];
  }
  return sum;
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

void load_dataset(const char *path, const char *label, int count, float train[][IMAGE_WIDTH*IMAGE_WIDTH*IMAGE_CHANNELS]){
  for (int i=0; i<count; i++){
    char filename[128] = {0};
    sprintf(filename, "%s/%s.%d.jpg", path, label, i);
    load_image(filename, train[i]);
  }
}

typedef struct{
  float a0[IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS];
  float w1[IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS];
  float b1;
  float a1;
  float z1;
} neural_network;

void nn_init(neural_network *nn){
  for(int i=0; i<IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS; i++){
    nn->w1[i] = (rand() / (float)RAND_MAX -0.5f) * 2.0;
  }  
  nn->b1 = (rand() / (float)RAND_MAX -0.5f) * 2.0;
}

float feed_forward(neural_network *nn, float *x){
  memcpy(nn->a0, x, IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS*sizeof(float));
  nn->z1 = dot_product(nn->w1, x, IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS) + nn->b1;
  nn->a1 = sigmoid(nn->z1);

  return nn->a1;
}

#define nn_loss(y, y_hat) (y-y_hat)*(y-y_hat);
// float nn_loss(float y, float y_hat){
//   return (y-y_hat) * (y-y_hat);
// }

void nn_gradient(neural_network *nn, float y_hat, neural_network *grad){
  // want dC/dw1 = dC/da1 * dz1/da1 * dz1/dw1
  float dC_da1 = 2* (nn->a1 - y_hat);
  float da1_dz1 = sigmoid(nn->a1) * (1-sigmoid(nn->a1));
  for (int i=0; i<IMAGE_WIDTH*IMAGE_WIDTH*IMAGE_CHANNELS; i++){
    float dz1_dw1 = nn->a0[i];
    grad->w1[i] = dC_da1 * da1_dz1 * dz1_dw1;
  }

  float dz1_db1 = 1;
  grad->b1 = dC_da1 * da1_dz1 * dz1_db1;
}

void nn_backward(neural_network *nn, neural_network *grad, float learning_rate){
  for (int i=0; i<IMAGE_WIDTH*IMAGE_WIDTH*IMAGE_CHANNELS; i++){
    nn->w1[i] -= grad->w1[i] * learning_rate;
  }
  nn->b1 -= grad->b1 * learning_rate;
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

  load_dataset("dvc/train/train","cat", TRAIN_COUNT, cat_train);
  load_dataset("dvc/train/train","dog", TRAIN_COUNT, dog_train);

  total = 2*TRAIN_COUNT;
  correct = compute_true_positive(&nn, cat_train, CAT_LABEL) + compute_true_positive(&nn, dog_train, DOG_LABEL);
  printf("%f\n", correct/(float)total);

  for (int i=0; i<EPOCHS; i++){
    learn(&nn, cat_train, CAT_LABEL, LEARNING_RATE);
    learn(&nn, dog_train, DOG_LABEL, LEARNING_RATE);

    float cat_loss = compute_loss(&nn, cat_train, CAT_LABEL);
    float dog_loss = compute_loss(&nn, dog_train, DOG_LABEL);
    float loss = (cat_loss+dog_loss)/2.0;
    printf("%f\n", loss);
  }

  total = 2*TRAIN_COUNT;
  correct = compute_true_positive(&nn, cat_train, CAT_LABEL) + compute_true_positive(&nn, dog_train, DOG_LABEL);
  printf("finale: %f\n", correct/(float)total);

  return 0;
}

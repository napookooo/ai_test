#include <stdio.h>
#include <time.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
  float weights_input[IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS];
  float bias_input;
} neural_network;

void nn_init(neural_network *nn){
  for(int i=0; i<IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS; i++){
    nn->weights_input[i] = (rand() / (float)RAND_MAX -0.5f) * 2.0;
  }  
  nn->bias_input = (rand() / (float)RAND_MAX -0.5f) * 2.0;
}

float feed_forward(const neural_network *nn, float *x){
  float z = dot_product(nn->weights_input, x, IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS) + nn->bias_input;

  float y = sigmoid(z);

  return y;
}

int main(int argc, char *argv[]){

  srand(time(NULL));

  neural_network nn;
  nn_init(&nn);

  load_dataset("dvc/train/train","cat", TRAIN_COUNT, cat_train);
  load_dataset("dvc/train/train","dog", TRAIN_COUNT, dog_train);

  int total = 2*TRAIN_COUNT;
  int correct = 0;
  for (int i=0; i<TRAIN_COUNT; i++){
    float prediction = feed_forward(&nn, cat_train[i]);

    if (prediction >= 0.5){
      correct++;
    }

    prediction = feed_forward(&nn, dog_train[i]);
    if (prediction <= 0.5){
      correct++;
    }
  }

  printf("%f\n", correct/(float)total);

  return 0;
}

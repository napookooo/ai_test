#include <stdio.h>
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

int main(int argc, char *argv[]){

  load_dataset("dvc/train/train","cat", TRAIN_COUNT, cat_train);
  load_dataset("dvc/train/train","dog", TRAIN_COUNT, dog_train);

  float x = 1.2;

  float w, b;
  float y = w*x + b;

  printf("%f", y);

  return 0;
}

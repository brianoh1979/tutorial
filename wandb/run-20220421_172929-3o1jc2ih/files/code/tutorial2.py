#!/usr/bin/env python
import wandb
#import tensorflow as tf
wandb.init(project="project2", entity="brianoh1979h")

"""
This Weights & Biases sample script trains a basic CNN on the
Fashion-MNIST dataset. It takes black and white images of clothing
and labels them as "pants", "belt", etc. This script is designed
to demonstrate the wandb integration with Keras.
"""
from tensorflow import keras 
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import random

#Import wandb libraries
import wandb
from wandb.keras import WandbCallback

# Initialize wandb and save hyperparameters
#wandb.init(project="project1", entity="brianoh1979h")
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128 
}

wandb.log({"loss": 0.314, "epoch": 5,
           "inputs": wandb.Image(inputs),
           "logits": wandb.Histogram(outputs),
           "captions": wandb.Html(captions)})

#config = wandb.config

with tf.Session() as sess:
  # ...
  wandb.tensorflow.log(tf.summary.merge_all())

wandb.log({"loss": loss})

# Optional
wandb.watch(model)

(X_train_orig, y_train_orig), (X_test, y_test) = fashion_mnist.load_data()

# Reducing the dataset size to 10,000 examples for faster train time
true = list(map(lambda x: True if random.random() < 0.167 else False, range(60000)))
ind = []
for i, x in enumerate(true):
    if x == True: ind.append(i)

X_train = X_train_orig[ind, :, :]
y_train = y_train_orig[ind]

img_width=28
img_height=28
labels =["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

#reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

sgd = SGD(decay=config.decay, momentum=config.momentum,
                            nesterov=True)

# build model
model = Sequential()
model.add(Conv2D(config.layer_1_size, (5, 5), activation='relu',
                            input_shape=(img_width, img_height,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(config.layer_2_size, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(config.dropout))
model.add(Flatten())
model.add(Dense(config.hidden_layer_size, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#Add Keras WandbCallback
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=config.epochs,
    callbacks=[WandbCallback(data_type="image", labels=labels)])

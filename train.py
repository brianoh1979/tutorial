#!/usr/bin/env python
import wandb
wandb.init(project="firstproject")

"""
Defines a simple CNN model on the fashion mnist dataset.
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils

import wandb
from wandb.keras import WandbCallback

# define the image width and height acc to the dataset
img_width=28
img_height=28

def train_cnn(args):
  # initialize wandb logging for the project
  wandb.init(project=args.project_name)
  # log all experimental args to wandb
  wandb.config.update(args)

  # load and prepare data
  (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
  labels=["T-shirt/top","Trouser","Pullover","Dress","Coat", "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

  # normalize the data
  X_train = X_train.astype('float32')
  X_train /= 255.
  X_test = X_test.astype('float32')
  X_test /= 255.

  # reshape input data
  X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
  X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

  # one hot encode outputs
  y_train = np_utils.to_categorical(y_train)
  y_test = np_utils.to_categorical(y_test)
  num_classes = y_test.shape[1] # = 10, as there are 10 classes in fashion mnist dataset

  # build model
  model = Sequential()
  model.add(Conv2D(args.L1_conv_size, (5, 5), activation='relu', input_shape=(img_width, img_height,1)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(args.L2_conv_size, (5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(args.dropout_mask))
  model.add(Flatten())
  model.add(Dense(args.hidden_size, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))

  adam = Adam(lr=args.learning_rate)

  # enable logging for validation examples
  val_generator = ImageDataGenerator()
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  model.fit(X_train, y_train, 
            validation_data=(X_test, y_test), 
            epochs=args.epochs,
            callbacks=[WandbCallback(data_type="image", 
                                    labels=labels, 
                                    generator=val_generator.flow(X_test, y_test, batch_size=32))])

  # save the trained model
  model.save(f"{args.model_name}.h5")

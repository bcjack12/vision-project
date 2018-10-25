from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.callbacks import Callback
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import keras
import subprocess
import os
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

config.batch_size = 32
config.num_epochs = 50
config.first_layer_convs = 32
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
config.dropout = 0.4
config.dense_layer_size = 128
config.img_width = 48
config.img_height = 48
config.epochs = 50

input_shape = (48, 48, 1)

def load_fer2013():
    if not os.path.exists("fer2013"):
        print("Downloading the face emotion dataset...")
        subprocess.check_output("curl -SL https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar | tar xz", shell=True)
    data = pd.read_csv("fer2013/fer2013.csv")
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = np.asarray(pixel_sequence.split(' '), dtype=np.uint8).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    val_faces = faces[int(len(faces) * 0.8):]
    val_emotions = emotions[int(len(faces) * 0.8):]
    train_faces = faces[:int(len(faces) * 0.8)]
    train_emotions = emotions[:int(len(faces) * 0.8)]
    
    return train_faces, train_emotions, val_faces, val_emotions

# loading dataset

train_faces, train_emotions, val_faces, val_emotions = load_fer2013()
num_samples, num_classes = train_emotions.shape

train_faces /= 255.
val_faces /= 255.

model = Sequential()
model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(48, 48,1),
    activation='relu', padding='valid'))#
model.add(MaxPooling2D(pool_size=(2, 2)))#
model.add(Dropout(0.25))

model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(48, 48,1),
    activation='relu', padding='valid'))#
model.add(MaxPooling2D(pool_size=(2, 2)))#
model.add(Dropout(0.25))

model.add(Conv2D(64,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(48, 48,1),
    activation='relu', padding='valid'))#
model.add(MaxPooling2D(pool_size=(2, 2)))#
model.add(Dropout(0.25))


model.add(Flatten(input_shape=input_shape))
model.add(Dense(8*64*7, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

#model.add(Dense(config.dense_layer_size, activation='relu'))#
#model.add(Dense(num_classes, activation="softmax"))
opt = keras.optimizers.Adam(lr=1*0.001) #, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#model.compile(optimizer='adam', loss='categorical_crossentropy',
#metrics=['accuracy'])

model.compile(optimizer=opt, loss='categorical_crossentropy',
metrics=['accuracy'])

model.fit(train_faces, train_emotions, batch_size=config.batch_size,
        epochs=config.num_epochs, verbose=1, callbacks=[
            WandbCallback(data_type="image", labels=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
        ], validation_data=(val_faces, val_emotions))


model.save("emotion.h5")




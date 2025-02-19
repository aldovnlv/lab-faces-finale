# https://github.com/adsoftsito/taller_itsz?tab=readme-ov-file
# https://github.com/JazaeloG/IABunnies/blob/master/Entrenamiento.py
# https://github.com/aldovnlv/caras_training/blob/main/flowers-model.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import base64
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

# Helper libraries
# import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import os
import pathlib

# Set the path of the input folder

dataset = "https://drive.usercontent.google.com/download?id=17kdDJE4GnYSuNDW_7i1r31n9ReWTnIhd&export=download&authuser=0&confirm=t&uuid=b2c1cebf-4c91-4c42-b6ab-0ba704158cec&at=AIrpjvP4J4FJzIIKkGZLlSB-Qqre%3A1738345923439"

directory = tf.keras.utils.get_file('flower_photos', origin=dataset, untar=True, cache_dir='.')
data = pathlib.Path(directory)

size = 150,150

images = []
labels = []
listaPersonas = os.listdir(data)

dataPath = f'{os.getcwd()}/datasets/flower_photos'

for nombrePersona in listaPersonas:
    rostrosPath = os.path.join(dataPath, nombrePersona)
    print(rostrosPath)
    for filename in os.listdir(rostrosPath):
        img_path = os.path.join(rostrosPath, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(img,size)
        images.append(im)
        labels.append(nombrePersona)

images = np.array(images)
labels = np.array(labels)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(listaPersonas), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train.reshape(-1, 150, 150, 1), y_train, epochs=100, validation_data=(X_test.reshape(-1, 150, 150, 1), y_test))

# Guarda el modelo
export_path = 'faces-model/1/'
tf.keras.models.save_model(model, os.path.join('./', export_path))
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

# Helper libraries
# import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import join

import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import base64

print(os.getcwd())

# Set the path of the input folder

dataset = "https://drive.usercontent.google.com/download?id=17kdDJE4GnYSuNDW_7i1r31n9ReWTnIhd&export=download&authuser=0&confirm=t&uuid=b2c1cebf-4c91-4c42-b6ab-0ba704158cec&at=AIrpjvP4J4FJzIIKkGZLlSB-Qqre%3A1738345923439"
directory = tf.keras.utils.get_file('flower_photos', origin=dataset, untar=True)
data = pathlib.Path(directory)
folders = os.listdir(data)
print(folders)

images = []
labels = []
listaPersonas = ['Daniela', 'Jazael']

dataPath = f'{os.getcwd()}/data'

print(dataPath)


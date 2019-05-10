# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import cv2
from pprint import pprint
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_labels[0])


print(tf.__version__)
tf.enable_eager_execution()

speedData = []

# with open("./data/train.txt", "r") as filehandle:  
#     for line in filehandle:
#         # remove linebreak which is the last character of the string
#         speed = line[:-1]

#         # add item to the list
#         speedData.append(speed)

train_images = []

img_raw = tf.read_file('./data/road-left/0-left.jpg')
img_tensor = tf.image.decode_jpeg(img_raw)

img_final = img_tensor/255.0

print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

print(img_final)

train_images.append(img_final)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50, 50)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, [1], epochs=5)

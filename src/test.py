from neural_net import NeuralNetwork
import numpy as np

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
from distutils.dir_util import copy_tree, remove_tree
from random import randint

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pathlib import Path

# Get directories
base_dir = './../data/Alzheimer_s Dataset/'
root_dir = "./../data/"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = os.path.join(root_dir, "work_dir")

os.mkdir(work_dir)

print("Testing directory contents:", os.listdir(test_dir))
print("Training directory contents:", os.listdir(train_dir))

classes = ['NonDemented',
           'VeryMildDemented',
           'MildDemented',
           'ModerateDemented']

image_size = 96
image_dim = (image_size, image_size)

# Performing Image Augmentation to have more data samples

zoom = [.99, 1.01]
brightness = [0.8, 1.2]
fill_mode = "constant"
data_format = "channels_last"

work_dr = ImageDataGenerator(rescale=1. / 255, brightness_range=brightness, zoom_range=zoom, fill_mode=fill_mode,
                             data_format=data_format)

train_gen = work_dr.flow_from_directory(directory=work_dir, target_size=image_dim, batch_size=6500, shuffle=True)

labels = dict(zip([0, 1, 2, 3], classes))

# get a batch of images
x, y = train_gen.next()

plt.figure(figsize=(12, 12))

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    idx = randint(0, 6400)
    plt.imshow(x[idx])
    plt.axis("off")
    plt.title("Class:{}".format(labels[np.argmax(y[idx])]))

# Retrieving the data from the ImageDataGenerator iterator

train_data, train_labels = train_gen.next()
print(train_data.shape, train_labels.shape)

# Performing over-sampling of the data, since the classes are imbalanced

sm = SMOTE(random_state=18)

train_data, train_labels = sm.fit_resample(train_data.reshape(-1, image_size * image_size * 3), train_labels)

train_data = train_data.reshape(-1, image_size, image_size, 3)

print(train_data.shape, train_labels.shape)

# Splitting the data into train, test, and validation sets

train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                    random_state=3)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                  random_state=3)

print(train_data)

layers_dims = [3, 6, 4]

nn = NeuralNetwork(layers_dims=layers_dims, activation="softmax")

# nn.fit(X, y)

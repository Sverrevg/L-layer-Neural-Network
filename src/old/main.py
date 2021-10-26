from neural_net import NeuralNet
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

"""
Prepare data:
"""
train_dataset = h5py.File('sample_data/train_catvnoncat.h5', "r")
train_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
train_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

test_dataset = h5py.File('sample_data/test_catvnoncat.h5', "r")
test_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
test_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

classes = np.array(test_dataset["list_classes"][:])  # the list of classes

train_y = train_y_orig.reshape((1, train_y_orig.shape[0]))
test_y = test_y_orig.reshape((1, test_y_orig.shape[0]))

# Example of a picture
index = 88
plt.imshow(train_x_orig[index])
plt.show()
print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.\n")

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")

# Reshape the training and test examples
# The "-1" makes reshape flatten the remaining dimensions
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape) + "\n")

"""
Prepare model:
"""
layers_dims = [12288, 20, 10, 7, 3, 1]  # 4-layer model
# Initialise the model with hyperparameters (override some defaults):
nn = NeuralNet(layers_dims, iterations=2000, print_cost=True)

nn.train(train_x, train_y)

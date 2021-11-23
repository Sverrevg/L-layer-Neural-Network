import numpy as np
import h5py
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from neural_net import NeuralNetwork


def load_data():
    train_dataset = h5py.File('./../data/cats/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('./../data/cats/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


X_train_origin, y_train, X_test_origin, y_test, classes = load_data()

# Example of a picture
index = 88
plt.imshow(X_train_origin[index])
plt.show()
print("y = " + str(y_train[0, index]) + ". It's a " + classes[y_train[0, index]].decode("utf-8") + " picture.\n")

# Explore your dataset:
image_size = X_train_origin.shape[1]
m_train = X_train_origin.shape[0]
m_test = X_test_origin.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(image_size) + ", " + str(image_size) + ", 3)")
print("train_x_orig shape: " + str(X_train_origin.shape))
print("train_y shape: " + str(y_train.shape))
print("test_x_orig shape: " + str(X_test_origin.shape))
print("test_y shape: " + str(y_test.shape) + "\n")

# Reshape the training and test examples:
# The "-1" makes reshape flatten the remaining dimensions:
X_train_flatten = X_train_origin.reshape(X_train_origin.shape[0], -1).T
X_test_flatten = X_test_origin.reshape(X_test_origin.shape[0], -1).T

# Standardize data to have feature values between 0 and 1:
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.

print("X_train's shape: " + str(X_train.shape))
print("X_test's shape: " + str(X_test.shape) + "\n")

# Constants
layers_dims = [12288, 15, 9, 5, 3, 1]  # 5-layer model

nn = NeuralNetwork(layers_dims, learning_rate=0.005, num_iterations=1800, activation="sigmoid")

nn.fit(X_train, y_train)

# Save parameters for later use:
nn.save_model()

# Plot cost over time:
fig, ax = plt.subplots()
ax.plot(nn.costs)
ax.set(xlabel='Iterations (x10)', ylabel='Cost', title='Cost over iterations')
plt.show()

# Test model:
pred_test = nn.test(X_test, y_test)

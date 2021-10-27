from PIL import Image
import numpy as np
import h5py
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork


def load_data():
    train_dataset = h5py.File('sample_data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('sample_data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

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
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape) + "\n")

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(
    train_x_orig.shape[0], -1).T  # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape) + "\n")

# Constants
layers_dims = [12288, 15, 9, 5, 1]  # 4-layer model

nn = NeuralNetwork(layers_dims, learning_rate=0.005, num_iterations=2500, print_cost=True)

costs = nn.train_model(train_x, train_y)

# Plot cost over time:
fig, ax = plt.subplots()
ax.plot(costs)
ax.set(xlabel='Iterations (x10)', ylabel='Cost', title='Cost over iterations')
plt.show()

# Test model:
pred_test = nn.test(test_x, test_y)

# Prepare image for prediction:
path = 'sample_data/cat.jpg'

image = np.array(Image.open(path).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
plt.show()

prediction = nn.predict(image)
print(f'Output: {prediction}')
print('Prediction: ' + 'cat' if prediction >= 0.5 else 'non-cat')

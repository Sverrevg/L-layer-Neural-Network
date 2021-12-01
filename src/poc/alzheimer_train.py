import os

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import randint

from neural_net import NeuralNetwork
from network_options import Activations, Loss

# Get directories
base_dir = '../data/Alzheimer_s Dataset/'
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = os.path.join(base_dir, "work_dir")

# Create work dir and copy files
# if os.path.exists(work_dir):
#     remove_tree(work_dir)
#
# os.mkdir(work_dir)
# copy_tree(train_dir, work_dir)
# copy_tree(test_dir, work_dir)

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
X_orig, y_orig = train_gen.next()

plt.figure(figsize=(12, 12))

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    idx = randint(0, 6400)
    plt.imshow(X_orig[idx])
    plt.axis("off")
    plt.title("Class:{}".format(labels[np.argmax(y_orig[idx])]))

plt.show()

# Retrieving the data from the ImageDataGenerator iterator

X, y = train_gen.next()

# Performing over-sampling of the data, since the classes are imbalanced
sm = SMOTE(random_state=18)

X_flatten, y = sm.fit_resample(X.reshape(-1, image_size * image_size * 3), y)

# Splitting the data into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_flatten, y, test_size=0.2, random_state=3)

# Swap axes because for some reason they are not in correct order:
X_train = np.swapaxes(X_train, 0, 1)
y_train = np.swapaxes(y_train, 0, 1)
X_test = np.swapaxes(X_test, 0, 1)
y_test = np.swapaxes(y_test, 0, 1)

# Truncate dataset:
# X_train = X_train[::, :2000]
# y_train = y_train[::, :2000]

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# Input dim is equal to image_width * image_height
input_dim = X_train.shape[0]

layers_dims = [input_dim, 64, 32, 4]

nn = NeuralNetwork(learning_rate=0.0075, layers_dims=layers_dims, activation=Activations.SIGMOID.value,
                   loss=Loss.CATEGORICAL.value,
                   num_iterations=2500)

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

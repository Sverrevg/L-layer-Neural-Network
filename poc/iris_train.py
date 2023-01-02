import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from neural_network.network_operations.activation import Activation
from neural_network.network_operations.loss import Loss
from neural_network.network_operations.optimizer import Optimizer
from neural_network.neural_network import NeuralNetwork

"""Import data"""
iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(iris, sep=',')

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes

print(df.head())
print(f'Shape: {df.shape}')

"""Prepare data"""
# Convert strings to ints:
df['label'] = pd.factorize(df['class'])[0]

# Creating instance of one-hot-encoder:
encoder = OneHotEncoder(handle_unknown='ignore')
# Perform one-hot encoding on 'label' column
encoder_df = pd.DataFrame(encoder.fit_transform(df[["label"]]).toarray())
# Merge one-hot encoded columns back with original DataFrame
final_df = df.join(encoder_df)
final_df.drop(["sepal_length", "sepal_width", "petal_length", "petal_width", "class", "label"], axis=1, inplace=True)

X = np.array(df[["sepal_length", "sepal_width", "petal_length", "petal_width"]])
y = np.array(final_df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Reshape datasets
X_train = np.swapaxes(X_train, 0, 1)
X_test = np.swapaxes(X_test, 0, 1)
y_train = np.swapaxes(y_train, 0, 1)
y_test = np.swapaxes(y_test, 0, 1)

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

""""Setup model"""

# Constants
input_dim = X_train.shape[0]
output_dim = df['class'].nunique()
layers_dims = [input_dim, 8, output_dim]  # 4-layer model

nn = NeuralNetwork(layers_dims, num_iterations=1500, activation=Activation.SIGMOID.value, loss=Loss.CATEGORICAL.value,
                   optimizer=Optimizer.SGDM)

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

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork

# Constants
layers_dims = [12288, 15, 9, 5, 1]  # 4-layer model

nn = NeuralNetwork(layers_dims, learning_rate=0.005, num_iterations=1800, print_cost=True)

nn.load_parameters()

# Prepare image for prediction:
path = 'data/cats/cat.jpg'
num_px = 64

image = np.array(Image.open(path).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
plt.show()

prediction = nn.predict(image)
print(f'Output: {prediction}')
print('Prediction: ' + 'cat' if prediction >= 0.5 else 'non-cat')

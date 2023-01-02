from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

from neural_network.neural_network import NeuralNetwork

nn = NeuralNetwork([0])

nn.load_model()

# Prepare image for prediction:
cat = 'data/cat.jpg'
beer = 'data/beer.jpg'
num_px = 64

image_1 = np.array(Image.open(cat).resize((num_px, num_px)))
plt.imshow(image_1)
image_1 = image_1 / 255.
image_1 = image_1.reshape((1, num_px * num_px * 3)).T

plt.show()

image_2 = np.array(Image.open(beer).resize((num_px, num_px)))
plt.imshow(image_2)
image_2 = image_2 / 255.
image_2 = image_2.reshape((1, num_px * num_px * 3)).T

plt.show()

startTime = time.time()

prediction_1 = np.squeeze(nn.predict(image_1))
print(f'First image prediction - {np.round(prediction_1, 3)}:', 'cat' if prediction_1 >= 0.5 else 'non-cat')

prediction_2 = np.squeeze(nn.predict(image_2))
print(f'Second image prediction - {np.round(prediction_2, 3)} :', 'cat' if prediction_2 >= 0.5 else 'non-cat')

execution_time = (time.time() - startTime)
print(f'Execution time in milliseconds: ' + str(round(execution_time * 1000, 2)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

from neural_net import NeuralNetwork


nn = NeuralNetwork(learning_rate=0.005, num_iterations=1800, print_cost=True)

nn.load_model()

# Prepare image for prediction:
cat = '../data/cats/cat.jpg'
beer = '../data/cats/beer.jpg'
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

count = 100000

for i in range(count):
    prediction_1 = np.squeeze(nn.predict(image_1))
    # print(f'First image prediction - {np.round(prediction_1, 3)}:', 'cat' if prediction_1 >= 0.5 else 'non-cat')

execution_time = (time.time() - startTime)
print(f'Execution time in seconds for {count}: ' + str(round(execution_time, 10)))

# prediction_2 = np.squeeze(nn.predict(image_2))
# print(f'Second image prediction - {np.round(prediction_2, 3)} :', 'cat' if prediction_2 >= 0.5 else 'non-cat')

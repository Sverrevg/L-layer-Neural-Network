from .neuron import Neuron


class Layer:
    def __init__(self, neuron_count):
        self.neurons = []
        for i in range(neuron_count):
            neuron = Neuron()
            self.neurons.append(neuron)

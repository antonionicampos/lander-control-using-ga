import numpy as np

class NN:

    relu = lambda self, x: np.maximum(x, 0)
    softmax = lambda self, x: np.exp(x) / np.sum(np.exp(x))

    def __init__(self, input_dim: int, output_dim: int, hidden_units: int = 64):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        self.initialize_weights()

    def initialize_weights(self):
        self.weights1 = np.random.normal(0., 1., size=(self.hidden_units, self.input_dim))
        self.bias1 = np.zeros((self.hidden_units, 1))
        self.weights2 = np.random.normal(0., 1, size=(self.output_dim, self.hidden_units))
        self.bias2 = np.zeros((self.output_dim, 1))

    def get_weights(self):
        return {
            'layer_0': {'weights': self.weights1, 'bias': self.bias1},
            'layer_1': {'weights': self.weights2, 'bias': self.bias2}
        }

    def set_weights(self, weights):
        pass

    def __call__(self, x: np.array):
        hidden = self.relu(self.weights1 @ x.reshape(-1, 1) + self.bias1)
        output = self.softmax(self.weights2 @ hidden + self.bias2)
        return np.argmax(output).item()

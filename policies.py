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

    def params_size(self):
        return self.get_weights().shape[0]

    def get_weights(self):
        params = (self.weights1, self.weights2, self.bias1, self.bias2)
        return np.concatenate(params, axis=None).reshape(-1, 1)

    def set_weights(self, genes):
        w1_size = self.input_dim*self.hidden_units
        w2_size = self.hidden_units*self.output_dim

        self.weights1 = genes[:w1_size].reshape(self.hidden_units, self.input_dim)
        self.weights2 = genes[w1_size:(w1_size + w2_size)].reshape(self.output_dim, self.hidden_units)
        self.bias1 = genes[-(self.output_dim + self.hidden_units):-self.output_dim].reshape(-1, 1)
        self.bias2 = genes[-self.output_dim:].reshape(-1, 1)

    def __call__(self, x: np.array):
        hidden = self.relu(self.weights1 @ x.reshape(-1, 1) + self.bias1)
        output = self.softmax(self.weights2 @ hidden + self.bias2)
        return np.argmax(output).item()

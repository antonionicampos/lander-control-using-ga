import numpy as np

from scipy.special import softmax


class NN:

    relu = lambda self, x: np.maximum(x, 0)

    def __init__(self, input_dim: int, output_dim: int, hidden_units: int = 64):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim

    def params_size(self):
        input_to_hidden = self.hidden_units * self.input_dim + self.hidden_units
        hidden_to_output = self.output_dim * self.hidden_units + self.output_dim
        return input_to_hidden + hidden_to_output

    def set_weights(self, genes):
        w1_size = self.input_dim*self.hidden_units
        w2_size = self.hidden_units*self.output_dim

        self.weights1 = genes[:w1_size].reshape(self.hidden_units, self.input_dim).copy()
        self.weights2 = genes[w1_size:(w1_size + w2_size)].reshape(self.output_dim, self.hidden_units).copy()
        self.bias1 = genes[-(self.output_dim + self.hidden_units):-self.output_dim].reshape(-1, 1).copy()
        self.bias2 = genes[-self.output_dim:].reshape(-1, 1).copy()

    def __call__(self, x: np.array):
        hidden = np.tanh(self.weights1 @ x.reshape(-1, 1).copy() + self.bias1)
        output = softmax(self.weights2 @ hidden + self.bias2)
        return np.argmax(output).item()

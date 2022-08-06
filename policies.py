import numpy as np

from scipy.special import softmax


class NN:

    def __init__(self, input_dim: int, output_dim: int, hidden_units: list = [64, 64]):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.all_units = [self.input_dim, *self.hidden_units, self.output_dim]

    def params_size(self):
        dim = 0
        for index in range(1, len(self.all_units)):
            dim += self.all_units[index] * self.all_units[index - 1] + self.all_units[index]
        return dim

    def set_weights(self, genes):
        self.params = []
        current_index = 0
        for index in range(1, len(self.all_units)):
            input_dim, output_dim = self.all_units[index - 1], self.all_units[index]
            bias_current_index = current_index + input_dim*output_dim
            weights = genes[current_index:current_index+input_dim*output_dim]
            biases = genes[bias_current_index:bias_current_index + output_dim]
            self.params.append({
                'weights': weights.reshape(output_dim, input_dim).copy(), 
                'biases': biases.reshape(-1, 1).copy()
            })
            current_index += input_dim*output_dim + output_dim

    def __call__(self, x: np.array):
        output = x.reshape(-1, 1).copy()
        for layer in self.params:
            weights, biases = layer['weights'], layer['biases']
            output = np.tanh(weights @ output + biases)
        output = softmax(output)
        return np.argmax(output).item()

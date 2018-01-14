# same as n_network except I tried to apply ony steve miller's backpropagation

from math import tanh
import numpy as np

class Network(object):
    
    def __init__(self, input_number, hidden_number, output_number):
        self.input_number = input_number
        self.hidden_number = hidden_number
        self.output_number = output_number

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return x * (1 - x)

    def _gradient(self, val):
        return 1 - val ** 2

    def create_connections(self, h_to_o_weight = 0.1):
        i_to_h_weight = 1 / self.input_number
        self.matrix_wih = []
        self.matrix_who = []

        np.random.seed(1)
        # fix the weights to the mind example to compare/debug: [[0.8,0.4,0.3],[0.2,0.9,0.5]
        self.matrix_wih = np.matrix([[0.8,0.4,0.3],[0.2,0.9,0.5]])
#         self.matrix_wih = np.matrix([[i_to_h_weight]* self.hidden_number]*self.input_number) # doesn't work
#         self.matrix_wih = 2 * np.random.random((self.input_number,self.hidden_number)) - 1

        # fix the weights to the mind example to compare/debug: [[0.3],[0.5],[0.9]]
        self.matrix_who = np.matrix([[0.3],[0.5],[0.9]])
        # self.matrix_who = np.matrix([[h_to_o_weight]* self.output_number]*self.hidden_number) # doesn't work
        # self.matrix_who = 2 * np.random.random((self.hidden_number, self.output_number)) - 1
        pass

    def feed_forward(self, input_mask):
        '''
        All the possible input outputs are set to 1.

        '''
        self.input_outputs = input_mask
        self.hidden_outputs = np.dot(input_mask, self.matrix_wih)
        self.hidden_outputs = self.sigmoid(self.hidden_outputs)

        self.output_ouputs = np.dot(self.hidden_outputs, self.matrix_who)
        self.output_ouputs = self.sigmoid(self.output_ouputs)

    def set_network(self, entries, outputs):
        pass

    def matrix_gradient(self, matrix):
        return np.vectorize(self.gradient)(matrix)

    def get_output_deltas(self, target):
        errors = target - self.output_ouputs
        #print('Errors %s'%errors)
        #vectorized_gradient = np.vectorize(self.gradient)
        output_gradients = self.matrix_gradient(self.output_ouputs)
        output_deltas = np.multiply(output_gradients, errors)
        return output_deltas

    def calculate_new_output_weight(self, output_deltas, learning_rate):
        # Delta weights = delta output sum / hidden layer results
        dow =  output_deltas / self.hidden_outputs
        # new weights a old weights + delta weights
        dowt = dow.T
        for col in range(self.matrix_who.shape[1]):
            self.matrix_who[:, col] = self.matrix_who[:, col] + dowt[:, col]
        pass

    def get_hidden_deltas(self, output_deltas):
        # Delta hidden sum = delta output sum / hidden-to-outer weights * S'(hidden sum)
        error_matrix = output_deltas / self.matrix_who
        #https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
        #error_matrix = np.divide(output_deltas, self.matrix_who.transpose())
        hidden_deltas = error_matrix.copy()
        for col in range(error_matrix.shape[1]):
            hidden_deltas[:, col] = error_matrix[:, col] * self.matrix_gradient(self.hidden_outputs)[:,0]
        return hidden_deltas

    def calculate_input_to_hidden_weight(self, hidden_deltas, learning_rate):
        dihw = hidden_deltas / self.input_outputs
        dihwt = dihw.T
        for row in range(self.matrix_wih.shape[0]):
            self.matrix_wih[row, : ] = self.matrix_wih[row, : ] + dihwt[row, : ]
            pass

    def back_propagate(self, target, learning_rate=0.5):
        output_deltas = self.get_output_deltas(target)
        hidden_deltas = self.get_hidden_deltas(output_deltas)
        # new weigths could be calculated before getting the hidden delta but we need
        # the old valueof the weights to calculate the hidden deltas
        self.calculate_new_output_weight(output_deltas,learning_rate)
        self.calculate_input_to_hidden_weight(hidden_deltas,learning_rate)

    def train(self, input_mask, target, learning_rate=0.5):
        self.feed_forward(input_mask)
        self.back_propagate(target, learning_rate)


if __name__ == '__main__':
    network = Network(input_number=3, hidden_number=1, output_number=3)
    network.create_connections()



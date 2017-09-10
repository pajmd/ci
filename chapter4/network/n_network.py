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
#        self.matrix_wih = np.matrix([[0.8,0.4,0.3],[0.2,0.9,0.5]])
        self.matrix_wih = np.matrix([[i_to_h_weight]* self.hidden_number]*self.input_number)
        self.matrix_wih = 2 * np.random.random((self.input_number,self.hidden_number)) - 1

#        self.matrix_who = np.matrix([[0.3],[0.5],[0.9]])
        self.matrix_who = np.matrix([[h_to_o_weight]* self.output_number]*self.hidden_number)
        self.matrix_who = 2 * np.random.random((self.hidden_number, self.output_number)) - 1
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

    def get_hidden_deltas(self, output_deltas):
        error_matrix = np.dot(output_deltas, self.matrix_who.transpose())
        #https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
        #error_matrix = np.divide(output_deltas, self.matrix_who.transpose())
        hidden_deltas = np.multiply(self.matrix_gradient(self.hidden_outputs),
                                    error_matrix)
        return hidden_deltas

    def update_hidden_output_weight(self, output_deltas, learning_rate):
        for k, h in enumerate(self.hidden_outputs):
            c = output_deltas * h
            self.matrix_who[k, :] =  self.matrix_who[k, :] + (c * learning_rate)
        # change = np.dot(self.hidden_outputs.transpose(), output_deltas)
        # #https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
        # #change = np.divide(output_deltas, self.hidden_outputs.transpose())
        # change = learning_rate * change
        # self.matrix_who = self.matrix_who + change

    def update_input_to_hidden_weight(self, hidden_deltas, learning_rate):
        for k, io in enumerate(self.input_outputs):
            c = hidden_deltas * io
            self.matrix_wih[k, :] =  self.matrix_wih[k, :] + (c * learning_rate)
        # change = np.dot(np.matrix(self.input_outputs).transpose(), hidden_deltas)
        # # https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
        # #change = np.divide(hidden_deltas, np.matrix(self.input_outputs).transpose())
        # change = learning_rate * change
        # self.matrix_wih = self.matrix_wih + change

    def back_propagate(self, target, learning_rate=0.5):
        output_deltas = self.get_output_deltas(target)
        hidden_deltas = self.get_hidden_deltas(output_deltas)
        self.update_hidden_output_weight(output_deltas,learning_rate)
        self.update_input_to_hidden_weight(hidden_deltas,learning_rate)

    def train(self, input_mask, target, learning_rate=0.5):
        self.feed_forward(input_mask)
        self.back_propagate(target, learning_rate)


if __name__ == '__main__':
    network = Network(input_number=3, hidden_number=1, output_number=3)
    network.create_connections()



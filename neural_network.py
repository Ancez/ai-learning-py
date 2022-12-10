import numpy as np

class NeuralNetwork:
  def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate):
    # initialize the number of input nodes
    self.input_nodes = num_inputs

    # initialize the number of hidden nodes
    self.hidden_nodes = num_hidden

    # initialize the number of output nodes
    self.output_nodes = num_outputs

    # initialize the network's weights and biases
    self.weights_input_to_hidden = np.random.normal(0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.input_nodes))
    self.weights_hidden_to_output = np.random.normal(0, self.output_nodes**-0.5, (self.output_nodes, self.hidden_nodes))
    self.biases_hidden = np.zeros((self.hidden_nodes, 1))
    self.biases_output = np.zeros((self.output_nodes, 1))

    # set the learning rate
    self.learning_rate = learning_rate

  def activation_function(self, x):
    # apply the sigmoid function
    return 1 / (1 + np.exp(-x))

  def forward_propagation(self, inputs):
    # compute the output of the hidden layer
    hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) + self.biases_hidden
    hidden_outputs = self.activation_function(hidden_inputs)

    # compute the output of the output layer
    output_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) + self.biases_output
    outputs = self.activation_function(output_inputs)

    return outputs, hidden_outputs

  def backpropagation(self, inputs, hidden_outputs, outputs, targets):
    # compute the error at the output layer
    output_errors = targets - outputs

    # compute the error at the hidden layer
    hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)

    # update the weights and biases using gradient descent
    self.weights_hidden_to_output += self.learning_rate * np.dot(output_errors, hidden_outputs.T)
    self.weights_input_to_hidden += self.learning_rate * np.dot(hidden_errors, inputs.T)
    self.biases_hidden += self.learning_rate * hidden_errors
    self.biases_output += self.learning_rate * output_errors

  def train(self, inputs, targets):
    # perform forward propagation to get the outputs and hidden outputs
    outputs, hidden_outputs = self.forward_propagation(inputs)

    # perform backpropagation to update the weights and biases
    self.backpropagation(inputs, hidden_outputs, outputs, targets)

  def predict(self, inputs):
    # perform forward propagation to get the outputs
    outputs, _ = self.forward_propagation(inputs)

    return outputs

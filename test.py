from neural_network import NeuralNetwork
import numpy as np

# initialize the neural network
nn = NeuralNetwork(num_inputs=2, num_hidden=3, num_outputs=1, learning_rate=0.1)

# define some training data
data = [
  (np.array([[1, 2]]).T, np.array([[0]])),
  (np.array([[3, 4]]).T, np.array([[1]])),
  (np.array([[5, 6]]).T, np.array([[1]]))
]

for inputs, targets in data:
  nn.train(inputs, targets)

# define some test data
test_data = [
  (np.array([[1, 2]]).T, np.array([[0]])),
  (np.array([[3, 4]]).T, np.array([[1]])),
  (np.array([[5, 6]]).T, np.array([[1]]))
]

# use the neural network to make predictions on the test data
for inputs, targets in test_data:
  prediction = nn.predict(inputs)
  print(prediction)

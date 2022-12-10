# Ai Learning Python
The `NeuralNetwork` class is a simple implementation of a feedforward neural network with a single hidden layer. The network can be trained using gradient descent and backpropagation to make predictions on new data.

## Installation
To use the `NeuralNetwork` class, simply clone the repository and import the `neural_network.py` module:

```
git clone https://github.com/Ancez/ai-learning-py.git

cd ai-learning-py

from neural_network import NeuralNetwork
```

Usage
To create a new `NeuralNetwork` instance, use the `NeuralNetwork` constructor and provide the following parameters:

- `num_inputs`: The number of input nodes in the network.
- `num_hidden`: The number of hidden nodes in the network.
- `num_outputs`: The number of output nodes in the network.
- `learning_rate`: The learning rate for gradient descent.

```
nn = NeuralNetwork(num_inputs=2, num_hidden=3, num_outputs=1, learning_rate=0.1)
```

To train the network on some data, use the train method and provide the input data and the corresponding target values:

```
data = [
  (np.array([[1, 2]]).T, np.array([[0]])),
  (np.array([[3, 4]]).T, np.array([[1]])),
  (np.array([[5, 6]]).T, np.array([[1]]))
]

for inputs, targets in data:
  nn.train(inputs, targets)
```

To use the network to make predictions on new data, use the predict method and provide the input data:

```
test_data = [
  (np.array([[1, 2]]).T, np.array([[0]])),
  (np.array([[3, 4]]).T, np.array([[1]])),
  (np.array([[5, 6]]).T, np.array([[1]]))
]

for inputs, targets in test_data:
  prediction = nn.predict(inputs)
  print(prediction)
```

## Contributing
If you find any bugs or have any suggestions for improvements, please open an issue or submit a pull request. Any contributions are welcome!

## License
This project is not licensed. Use at your own risk.
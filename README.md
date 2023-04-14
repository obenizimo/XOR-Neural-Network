# XOR-Neural-Network
This is a simple neural network implementation in C# that solves the XOR problem using backpropagation.

Neural Network Architecture
The neural network has the following architecture:

Input layer: 2 neurons
Hidden layer: 8 neurons
Output layer: 1 neuron
Each neuron has a bias and a list of weights that are randomly initialized between -1 and 1.

The activation function used in this implementation is the sigmoid function:

f(x) = 1 / (1 + exp(-x))


Training
The neural network is trained using backpropagation with a learning rate of 0.1. The training data consists of four input-output pairs for the XOR problem:

0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0

The training is performed for 50,000 epochs, where each epoch consists of iterating through all the input-output pairs and updating the weights using backpropagation.

Testing
After the training is complete, the neural network is tested on the same input-output pairs to evaluate its performance.

References
This implementation is based on the following resources:

Neural networks and deep learning by Michael Nielsen
A Simple Neural Network in C# by Luis Vieira 

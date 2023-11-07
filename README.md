# Neural-Network-on-Boston-Housing

A simple neural network is built to predict house values using the California Housing dataset. The dataset features 8 attributes, with the last column representing house values. The data is normalized to have a mean of 0 and a standard deviation of 1, which is essential for better model convergence.

The neural network architecture consists of three layers:

Input Layer: 8 neurons (corresponding to the dataset's features).
Hidden Layer 1: 16 neurons with a specified activation function.
Hidden Layer 2: 32 neurons with a specified activation function.
Output Layer: 1 neuron for house value prediction.

Two different activation functions will be used for separate experiments: ReLU (Rectified Linear Unit) and TanH (Hyperbolic Tangent). The objective is to find the optimal learning rate for the Stochastic Gradient Descent (SGD) optimizer for each activation function.

The network is trained for 20 epochs, involving forward and backward propagation to update weights and biases. After training, the final performance is evaluated, including metrics like loss and mean squared error for regression.

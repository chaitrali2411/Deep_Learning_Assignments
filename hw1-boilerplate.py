import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress visualization

# Small XOR-like dataset (Input features and labels)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [1]])

# ReLU Activation Function
def relu_fn(x):
    return np.maximum(0, x)  # Applies ReLU element-wise

# Derivative of ReLU Function
def relu_derivative_fn(x):
    return (x > 0).astype(float)  # Gradient is 1 for positive values, 0 otherwise

# Sigmoid Activation Function
def sigmoid_fn(x):
    return 1 / (1 + np.exp(-x))  # Standard sigmoid function

# Derivative of Sigmoid Function
def sigmoid_derivative_fn(x):
    return x * (1 - x)  # Gradient of sigmoid

# Xavier Initialization for stable weight initialization
np.random.seed(42)  # Ensuring reproducibility
hidden_neurons = 2  # Number of neurons in the hidden layer
input_neurons = 2  # Number of input features
output_neurons = 1  # Single output neuron for binary classification

# Initialize weights and biases with small random values
weights_input_hidden = np.random.randn(input_neurons, hidden_neurons) * np.sqrt(1 / input_neurons)
bias_hidden = np.ones((1, hidden_neurons)) * 0.01  # Small bias initialization

weights_hidden_output = np.random.randn(hidden_neurons, output_neurons) * np.sqrt(1 / hidden_neurons)
bias_output = np.ones((1, output_neurons)) * 0.01  # Small bias initialization

# Momentum parameters (initial velocities for weights and biases)
velocity_w1 = np.zeros_like(weights_input_hidden)
velocity_b1 = np.zeros_like(bias_hidden)
velocity_w2 = np.zeros_like(weights_hidden_output)
velocity_b2 = np.zeros_like(bias_output)
momentum_factor = 0.9  # Momentum coefficient

# Forward Propagation
def forward_pass(data, w1, b1, w2, b2):
    """
    Computes the forward propagation through the neural network.
    """
    hidden_linear = np.dot(data, w1) + b1  # Weighted sum for hidden layer
    hidden_activation = relu_fn(hidden_linear)  # Applying ReLU activation
    output_linear = np.dot(hidden_activation, w2) + b2  # Weighted sum for output layer
    output_activation = sigmoid_fn(output_linear)  # Applying Sigmoid activation
    return hidden_linear, hidden_activation, output_linear, output_activation

# Backpropagation
def backward_pass(data, labels, hidden_linear, hidden_activation, output_linear, output_activation, w2):
    """
    Computes gradients for weight and bias updates using backpropagation.
    """
    samples = data.shape[0]  # Number of training samples

    # Compute loss derivative (Mean Squared Error Loss)
    d_output = (output_activation - labels) * sigmoid_derivative_fn(output_activation)
    dw2 = np.dot(hidden_activation.T, d_output) / samples  # Gradient for weights in output layer
    db2 = np.sum(d_output, axis=0, keepdims=True) / samples  # Gradient for biases in output layer

    d_hidden = np.dot(d_output, w2.T)  # Propagate error to hidden layer
    d_hidden_linear = d_hidden * relu_derivative_fn(hidden_linear)  # Applying ReLU derivative
    dw1 = np.dot(data.T, d_hidden_linear) / samples  # Gradient for weights in hidden layer
    db1 = np.sum(d_hidden_linear, axis=0, keepdims=True) / samples  # Gradient for biases in hidden layer

    return dw1, db1, dw2, db2

# Update Weights using Momentum
def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    """
    Updates weights and biases using Momentum-based Gradient Descent.
    """
    global velocity_w1, velocity_b1, velocity_w2, velocity_b2

    velocity_w1 = momentum_factor * velocity_w1 + (1 - momentum_factor) * dw1
    w1 -= lr * velocity_w1

    velocity_b1 = momentum_factor * velocity_b1 + (1 - momentum_factor) * db1
    b1 -= lr * velocity_b1

    velocity_w2 = momentum_factor * velocity_w2 + (1 - momentum_factor) * dw2
    w2 -= lr * velocity_w2

    velocity_b2 = momentum_factor * velocity_b2 + (1 - momentum_factor) * db2
    b2 -= lr * velocity_b2

    return w1, b1, w2, b2

# Training Hyperparameters
learning_rate = 0.05  # Step size for weight updates
epochs = 10000  # Total iterations over the dataset
loss_history = []  # Store loss values for visualization

# Training Loop
for epoch in tqdm(range(epochs)):
    # Forward propagation
    z_hidden, a_hidden, z_output, a_output = forward_pass(inputs, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
    
    # Compute Loss (Mean Squared Error)
    loss = np.mean((a_output - targets) ** 2)
    loss_history.append(loss)
    
    # Backward propagation
    dw1, db1, dw2, db2 = backward_pass(inputs, targets, z_hidden, a_hidden, z_output, a_output, weights_hidden_output)

    # Update weights and biases
    weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = update_parameters(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, dw1, db1, dw2, db2, learning_rate)

# Plot Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(loss_history, label="Loss over epochs", color='b')
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

# Generate grid for decision boundary visualization
x1_vals = np.linspace(-0.5, 1.5, 100)
x2_vals = np.linspace(-0.5, 1.5, 100)
xx1, xx2 = np.meshgrid(x1_vals, x2_vals)
grid_points = np.c_[xx1.ravel(), xx2.ravel()]

# Predict on the grid
_, _, _, grid_predictions = forward_pass(grid_points, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
grid_predictions = grid_predictions.reshape(xx1.shape)

# Plot Decision Boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, grid_predictions, levels=[0, 0.5, 1], alpha=0.6, cmap="coolwarm")
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets.ravel(), edgecolors="k", cmap="coolwarm", s=100, label="Training Data")
plt.title("Decision Boundary for Binary Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Prediction Confidence")
plt.legend()
plt.show()

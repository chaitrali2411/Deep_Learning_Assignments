import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress visualization

# Small XOR-like dataset (Input features and labels)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [1]])

# Activation Functions
def relu_fn(x):
    return np.maximum(0, x)

def relu_derivative_fn(x):
    return (x > 0).astype(float)

def sigmoid_fn(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative_fn(x):
    return x * (1 - x)

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

# Function to train and evaluate the model
def train_model(hidden_neurons, learning_rate, epochs=10000):
    np.random.seed(42)
    input_neurons = 2
    output_neurons = 1
    
    # Initialize weights and biases
    weights_input_hidden = np.random.randn(input_neurons, hidden_neurons) * np.sqrt(1 / input_neurons)
    bias_hidden = np.ones((1, hidden_neurons)) * 0.01
    
    weights_hidden_output = np.random.randn(hidden_neurons, output_neurons) * np.sqrt(1 / hidden_neurons)
    bias_output = np.ones((1, output_neurons)) * 0.01
    
    loss_history = []
    for epoch in tqdm(range(epochs)):
        # Forward Pass
        hidden_linear, hidden_activation, output_linear, output_activation = forward_pass(inputs, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
        
        # Compute Loss (MSE)
        loss = np.mean((output_activation - targets) ** 2)
        loss_history.append(loss)
        
        # Backpropagation
        d_output = (output_activation - targets) * sigmoid_derivative_fn(output_activation)
        dw2 = np.dot(hidden_activation.T, d_output) / inputs.shape[0]
        db2 = np.sum(d_output, axis=0, keepdims=True) / inputs.shape[0]
        
        d_hidden = np.dot(d_output, weights_hidden_output.T) * relu_derivative_fn(hidden_linear)
        dw1 = np.dot(inputs.T, d_hidden) / inputs.shape[0]
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / inputs.shape[0]
        
        # Update Weights
        weights_input_hidden -= learning_rate * dw1
        bias_hidden -= learning_rate * db1
        weights_hidden_output -= learning_rate * dw2
        bias_output -= learning_rate * db2
    
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, loss_history

# Function to visualize decision boundary
def plot_decision_boundary(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, title):
    x1_vals = np.linspace(-0.5, 1.5, 100)
    x2_vals = np.linspace(-0.5, 1.5, 100)
    xx1, xx2 = np.meshgrid(x1_vals, x2_vals)
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    
    _, hidden_activation, _, grid_predictions = forward_pass(grid_points, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
    grid_predictions = grid_predictions.reshape(xx1.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx1, xx2, grid_predictions, levels=[0, 0.5, 1], alpha=0.6, cmap="coolwarm")
    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets.ravel(), edgecolors="k", cmap="coolwarm", s=100)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Prediction Confidence")
    plt.show()

# Hyperparameter Tuning Experiments
hidden_sizes = [2, 4, 8]
learning_rates = [0.01, 0.05, 0.1]

for hidden_neurons in hidden_sizes:
    for lr in learning_rates:
        print(f"Training with {hidden_neurons} hidden neurons and learning rate {lr}")
        w1, b1, w2, b2, loss_hist = train_model(hidden_neurons, lr)
        
        # Plot Loss Curve
        plt.figure(figsize=(8, 6))
        plt.plot(loss_hist, label=f"LR: {lr}, Neurons: {hidden_neurons}", color='b')
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.show()
        
        # Plot Decision Boundary
        plot_decision_boundary(w1, b1, w2, b2, f"Decision Boundary (Neurons: {hidden_neurons}, LR: {lr})")

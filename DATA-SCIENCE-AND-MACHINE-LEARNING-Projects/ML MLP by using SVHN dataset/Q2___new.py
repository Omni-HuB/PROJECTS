
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def linear(x):
    return x

def linear_derivative(x):
    return 1

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

# Weight initialization functions
def zero_init(shape):
    return np.zeros(shape)

def random_init(shape):
    return np.random.randn(*shape)

def normal_init(shape):
    return np.random.normal(0, 1, shape)

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate, activation_func, activation_derivative, weight_init_func, num_epochs, batch_size):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        self.weight_init_func = weight_init_func
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Initialize weights and biases
        self.weights, self.biases = self.initialize_weights()

    def initialize_weights(self):
        weights = [self.weight_init_func((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(len(self.layer_sizes)-1)]
        biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(len(self.layer_sizes)-1)]
        return weights, biases

    def forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.layer_sizes) - 2):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            A = self.activation_func(Z)
            activations.append(A)

        # Last layer with softmax activation
        Z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        A = softmax(Z)
        activations.append(A)

        return activations

    def backward_propagation(self, X, Y, activations):
        m = X.shape[0]
        gradients = []

        # Calculate gradient at the last layer
        dZ = activations[-1] - Y
        dW = np.dot(activations[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        gradients.append((dW, db))

        # Backpropagate through hidden layers
        for i in range(len(self.layer_sizes) - 3, -1, -1):
            dA = np.dot(dZ, self.weights[i+1].T)
            dZ = dA * self.activation_derivative(activations[i+1])
            dW = np.dot(activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            gradients.insert(0, (dW, db))

        return gradients

    def update_weights(self, gradients):
        for i in range(len(self.layer_sizes) - 1):
            self.weights[i] -= self.learning_rate * gradients[i][0]
            self.biases[i] -= self.learning_rate * gradients[i][1]

    def compute_loss(self, Y_pred, Y):
        m = Y.shape[0]
        return -np.sum(Y * np.log(Y_pred)) / m

    def fit(self, X, Y, validation_data=None):
        history = {'loss': [], 'val_loss': []}

        for epoch in range(self.num_epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                Y_batch = Y[i:i+self.batch_size]

                # Forward propagation
                activations = self.forward_propagation(X_batch)

                # Compute loss
                loss = self.compute_loss(activations[-1], Y_batch)
                history['loss'].append(loss)

                # Backward propagation
                gradients = self.backward_propagation(X_batch, Y_batch, activations)

                # Update weights and biases
                self.update_weights(gradients)

            # Validation loss
            if validation_data:
                X_val, Y_val = validation_data
                val_activations = self.forward_propagation(X_val)
                val_loss = self.compute_loss(val_activations[-1], Y_val)
                history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

        return history

    def predict(self, X):
        activations = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)

    def predict_proba(self, X):
        activations = self.forward_propagation(X)
        return activations[-1]

def plot_loss(history, activation_func):
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss vs. Epochs ({activation_func} Activation)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    # Load MNIST dataset
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    # Preprocess data
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    # Normalize pixel values to be between 0 and 1
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Convert labels to one-hot encoding
    Y_train_one_hot = np.eye(10)[Y_train]

    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train_one_hot, test_size=0.2, random_state=42)

    # Configure neural network
    layer_sizes = [X_train.shape[1], 256, 128, 64, 32, 10]  # Added output layer size
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 128

    activation_functions = [sigmoid, tanh, relu, linear, softmax]
    activation_derivatives = [sigmoid_derivative, tanh_derivative, relu_derivative, linear_derivative, softmax_derivative]
    weight_init_functions = [zero_init, random_init, normal_init]

    for activation_func, activation_derivative in zip(activation_functions, activation_derivatives):
        for weight_init_func in weight_init_functions:
            # Create and train the neural network
            nn = NeuralNetwork(layer_sizes, learning_rate, activation_func, activation_derivative, weight_init_func, num_epochs, batch_size)
            history = nn.fit(X_train, Y_train, validation_data=(X_val, Y_val))

            # Plot training and validation loss
            plot_loss(history, activation_func.__name__)

            # Evaluate and print accuracy on the test set
            Y_pred = nn.predict(X_test)
            test_accuracy = accuracy_score(Y_test, Y_pred)
            print(f'Test Accuracy ({activation_func.__name__} Activation, {weight_init_func.__name__} Weight Init): {test_accuracy:.4f}')

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical



def tanh_derivative(x):
    return 1 - np.square(x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def ini_weights_zeros(input_size, output_size):
    return np.zeros((input_size, output_size))

def ini_weights_random(input_size, output_size):
    return np.random.rand(input_size, output_size)

def ini_weights_normal(input_size, output_size):
    return np.random.normal(0, 1, (input_size, output_size))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)





class NeuralNetwork:
    def __init__(self, N, layer_sizes, lr, activation_fn, weight_init_fn, num_epochs, batch_size):
        self.N = N
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.activation_fn = activation_fn
        self.weight_init_fn = weight_init_fn
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        self.activations = []
        self.losses = []

    def initialize_weights(self, input_size, output_size):
        if self.weight_init_fn == "zero":
            return ini_weights_zeros(input_size, output_size)
        elif self.weight_init_fn == "random":
            return ini_weights_random(input_size, output_size)
        elif self.weight_init_fn == "normal":
            return ini_weights_normal(input_size, output_size)

    def initialize_activation_functions(self):
        self.activation_functions = [self.activation_fn] * (self.N - 1) + [softmax]
        self.activation_derivatives = [sigmoid_derivative, tanh_derivative, relu_derivative,
                                      leaky_relu_derivative, linear_derivative]

    def forward_pass(self, X):
        self.activations = []
        input_data = X
        self.activations.append(input_data)

        for i in range(self.N - 1):
            z = np.dot(input_data, self.weights[i]) + self.biases[i]
            activation = self.activation_functions[i](z)
            self.activations.append(activation)
            input_data = activation

        return input_data

    def backward_pass(self, X, Y, predictions):
        errors = Y - predictions
        delta = errors

        for i in range(self.N - 2, -1, -1):
            gradient = self.activation_derivatives[i](self.activations[i + 1])
            delta = delta * gradient
            self.weights[i] += self.lr * np.dot(self.activations[i].T, delta)
            self.biases[i] += self.lr * np.sum(delta, axis=0, keepdims=True)
            delta = np.dot(delta, self.weights[i].T)

    def fit(self, X, Y, X_val, Y_val):
        self.initialize_activation_functions()

        for epoch in range(self.num_epochs):
            for i in range(0, X.shape[0], self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_Y = Y[i:i + self.batch_size]

                predictions = self.forward_pass(batch_X)
                self.backward_pass(batch_X, batch_Y, predictions)

            # Validation loss calculation
            val_predictions = self.forward_pass(X_val)
            val_loss = -np.mean(Y_val * np.log(val_predictions + 1e-8))
            self.losses.append(val_loss)
            print(f"Epoch: {epoch + 1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}")

    def predict_proba(self, X):
        return self.forward_pass(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, X, Y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == np.argmax(Y, axis=1))
        return accuracy
    



def load_and_preprocess_mnist():
    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_val = X_val.reshape(X_val.shape[0], -1) / 255.0

    y_train = to_categorical(y_train, num_classes=10)
    y_val = to_categorical(y_val, num_classes=10)

    return X_train, y_train, X_val, y_val




def train_and_plot(X_train, y_train, X_val, y_val, layer_sizes, lr, activation_fn, weight_init_fn, num_epochs, batch_size):
    model = NeuralNetwork(len(layer_sizes), layer_sizes, lr, activation_fn, weight_init_fn, num_epochs, batch_size)
    model.fit(X_train, y_train, X_val, y_val)


    plt.plot(range(num_epochs), model.losses, label=activation_fn.__name__)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_and_preprocess_mnist()

    
    hidden_layers = 4
    layer_sizes = [256, 128, 64, 32]
    epochs = 100
    batch_size = 128

    
    activation_functions = [tanh, relu, leaky_relu, linear]

    for activation_fn in activation_functions:
        train_and_plot(X_train, y_train, X_val, y_val, layer_sizes, lr=0.01, activation_fn=activation_fn,
                       weight_init_fn="random", num_epochs=epochs, batch_size=batch_size)

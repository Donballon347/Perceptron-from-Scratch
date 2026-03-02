import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_features, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(n_features)
        self.bias = 0.5
        self.history = []
        self.epoch_loss = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _cross_entropy(self, target, prediction):
        # Clip prediction to be between a tiny number and almost 1
        prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
        return -(target * np.log10(prediction) + (1 - target) * np.log10(1 - prediction))

    def _get_weighted_sum(self, feature, weights, bias):
        return np.dot(feature, weights) + bias

    def train(self, X, y):
        # Store initial state
        self.history.append((self.weights.copy(), self.bias))

        for e in range(self.epochs):
            losses = []
            for feature, target in zip(X, y):
                # Forward Pass
                w_sum = self._get_weighted_sum(feature, self.weights, self.bias)
                prediction = self._sigmoid(w_sum)
                
                # Calculate Loss
                losses.append(self._cross_entropy(target, prediction))

                # Gradient Descent Update (Vectorized)
                error = target - prediction
                self.weights += self.lr * error * feature
                self.bias += self.lr * error
            
            self.epoch_loss.append(np.mean(losses))
            self.history.append((self.weights.copy(), self.bias))
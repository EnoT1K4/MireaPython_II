import math

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier


def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return math.tanh(x)


def ReLU(x):
    return max(0, x)


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

    def feedTan(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return tanh(total)

    def feedRE(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return ReLU(total)


def task1():
    class OurNeuralNetwork:

        def __init__(self):
            weights = np.array([0.5, 0.5, 0.5])
            bias = 0

            # Используем класс Neuron из предыдущего раздела
            self.h1 = Neuron(weights, bias)
            self.h2 = Neuron(weights, bias)
            self.h3 = Neuron(weights, bias)
            self.o1 = Neuron(weights, bias)

        def feedforward(self, x):
            out_h1 = self.h1.feedforward(x)
            out_h2 = self.h2.feedforward(x)
            out_h3 = self.h3.feedforward(x)

            out_o1 = self.o1.feedforward(np.array([out_h1, out_h2, out_h3]))

            return out_o1

    network = OurNeuralNetwork()
    x = np.array([1, 2, 3])
    print(network.feedforward(x))


def task2():
    class OurNeuralNetwork:

        def __init__(self):
            weights = np.array([1, 0])
            bias = 1

            # Используем класс Neuron из предыдущего раздела
            self.h1 = Neuron(weights, bias)
            self.h2 = Neuron(weights, bias)
            self.o1 = Neuron(weights, bias)
            self.o2 = Neuron(weights, bias)

        def feedforward(self, x):
            out_h1 = self.h1.feedforward(x)
            out_h2 = self.h2.feedforward(x)

            out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
            out_o2 = self.o2.feedforward(np.array([out_h2, out_h1]))

            return out_o1, out_o2

    network = OurNeuralNetwork()
    x = np.array([2, 3])
    print(network.feedforward(x))


def task3():
    digits = load_iris()
    X_digits, Y_digits = digits.data, digits.target
    X_train, X_test, Y_train, Y_test = train_test_split(X_digits, Y_digits, train_size=0.80, test_size=0.20,
                                                        stratify=Y_digits, random_state=123)
    mlp_classifier = MLPClassifier(random_state=123)
    mlp_classifier.fit(X_train, Y_train)
    Y_preds = mlp_classifier.predict(X_test)
    print(Y_preds[:15])
    print('Test Accuracy: %3.f' % mlp_classifier.score(X_test, Y_test))
    print('Train Accuracy: %3.f' % mlp_classifier.score(X_train, Y_train))
    print('Loss:', mlp_classifier.loss_)


def task4():
    digits = load_iris()
    X_digits, Y_digits = digits.data, digits.target
    X_train, X_test, Y_train, Y_test = train_test_split(X_digits, Y_digits, train_size=0.80, test_size=0.20,
                                                        stratify=Y_digits, random_state=123)
    mlp_regressor = MLPRegressor(random_state=123)
    mlp_regressor.fit(X_train, Y_train)
    Y_preds = mlp_regressor.predict(X_test)
    print(Y_preds[:10])
    print('Train: %.3f'%mlp_regressor.score(X_train,Y_train))
    print('Loss', mlp_regressor.loss_)

print('Enter num task 1 or 2 or 3 or 4')
inp = int(input())
while inp != 0:
    if inp == 1:
        task1()
        break
    elif inp == 2:
        task2()
        break
    elif inp == 3:
        task3()
        break
    elif inp == 4:
        task4()
        break
    else:
        print("Check input")
        break

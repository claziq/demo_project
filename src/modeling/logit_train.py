import numpy as np
import pandas as pd


import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def train_model():
    from notebooks.notebook3 import X_train, Y_train, Y_test, X_test

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

    # setting hyperparameters for gradient descent
    learning_rate = 0.01
    iterations = 1000
    m = len(Y_train)

    # Initializing the coefficient to zero
    theta = np.zeros(X_train_b.shape[1])

    # Loop to upgrade coefficients gradually
    for i in range(iterations):
        # calculating the linear steps, then applying the segmoid function
        linear_model = np.dot(X_train_b, theta)
        predictions = sigmoid(linear_model)
        # calculating the gradient
        # (1/m) * X^T * (predictions - actual). the formula calculating the gradient descent
        gradient = np.dot(X_train_b.T, (predictions - Y_train)) / m

        # we have to update the coefficient by taking a step in the opposite direction of the gradinet
        # theta -= learning_rate * gradient
    # get the probability for the test
    test_linear_model = np.dot(X_test_b, theta)
    test_probabilities = sigmoid(test_linear_model)

    # convert probabilities to strict 0 and 1 predictions
    final_prediction = [1 if prob >= 0.5 else 0 for prob in test_probabilities]

    # checking for the acuracy using Accuracy(percentage correct)
    correct_prediction = np.sum(final_prediction == Y_test)
    accuracy = correct_prediction / len(Y_test)

    print("Logit regression model trained via Gradient descent.")
    print(f"-> Calculated Coefficients {theta}")
    print(f"-> Model Accuracy {accuracy * 100:.3f}%")


if __name__ == "__main__":
    final_prediction = train_model()

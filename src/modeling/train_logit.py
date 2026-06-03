import numpy as np
from src.data_processing import data_cleaning


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def main():
    X_test_scaled, X_train_scaled, Y_train, Y_test = data_cleaning()

    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    X_train_b = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]
    X_test_b = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]

    # setting hyperparameters for gradient descent
    learning_rate = 0.001
    iterations = 5000
    Y_train = Y_train.reshape(-1, 1)
    m = len(Y_train)

    theta = np.zeros((X_train_b.shape[1], 1))
    # Loop to upgrade coefficients gradually
    for _ in range(iterations):
        # calculating the linear steps, then applying the segmoid function
        linear_model = np.dot(X_train_b, theta)
        predictions = sigmoid(linear_model)
        # calculating the gradient
        # (1/m) * X^T * (predictions - actual). the formula calculating the gradient descent
        gradient = np.dot(X_train_b.T, (predictions - Y_train)) / m

        # we have to update the coefficient by taking a step in the opposite direction of the gradinet
        theta -= learning_rate * gradient
        # get the probability for the test
    test_linear_model = np.dot(X_test_b, theta)
    test_probabilities = sigmoid(test_linear_model)

    # convert probabilities to strict 0 and 1 predictions
    final_prediction = (test_probabilities >= 0.5).astype(int)

    # checking for the acuracy using Accuracy(percentage correct)
    accuracy = np.mean(final_prediction == Y_test)
    print(f"-> Calculated Coefficients {theta}")
    print(f"-> Model Accuracy {accuracy * 100:.3f}%")


if __name__ == "__main__":
    main()

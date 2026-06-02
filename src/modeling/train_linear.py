import numpy as np
from src.data_processing import data_cleaning


def main():
    X_train_scaled, X_test_scaled, Y_train, Y_test = data_cleaning()

    X_train_b = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]
    X_test_b = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]

    # Fitting the model to normal equation
    best_fit_theta = np.linalg.inv(X_train_b.T.dot(
        X_train_b)).dot(X_train_b.T).dot(Y_train)

    # pridicting with the model by multiplying X_test_b with best_fit_theta
    predictions = X_test_b.dot(best_fit_theta)

    # Calculating the mean square error
    mse = np.mean((Y_test - predictions) ** 2)

    print(f"-> Mean Square Error: {mse:.3f}")


if __name__ "__main__":
    main()

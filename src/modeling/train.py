import numpy as np
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def train_model():
    # isort: skip # pylint: disable=wrong-import-position
    from notebooks.notebook3 import X_train_b, Y_train
    best_fit_theta = np.linalg.inv(X_train_b.T.dot(
        X_train_b)).dot(X_train_b.T).dot(Y_train)
    print("Model trained succesfully")
    return best_fit_theta


if __name__ == "__main__":
    best_fit_theta = train_model()

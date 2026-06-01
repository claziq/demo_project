import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def predict():
    from notebooks.notebook3 import X_test_b, best_fit_theta
    predictions = X_test_b.dot(best_fit_theta)


if __name__ == "__main__":
    predicts = predict()

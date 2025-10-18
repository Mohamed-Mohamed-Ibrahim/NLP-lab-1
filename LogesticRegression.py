import numpy as np
import pandas as pd
# from matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


class LogisticRegressionCustom():
    def __init__(self, lr=0.001, epochs=100, threshold=0.5, logging=False):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.logging = logging

    def sigmoid(self, z):
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    def score(self, X, y):
        predictions = self.sigmoid(X.dot(self.weights) + self.bias)
        eps = 1e-10
        predictions = np.clip(predictions, eps, 1 - eps)
        loss = -np.mean(y * np.log2(predictions) + (1 - y) * np.log2(1 - predictions))
        return loss
    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            predictions = self.sigmoid(X.dot(self.weights) + self.bias)
            loss = predictions - y
            self.weights -= self.lr * (X.T.dot(loss) / n_samples)
            self.bias -= self.lr * loss.mean()

            if self.logging and epoch % 10 == 0:
                print("epoch:", epoch, "loss:", self.score(X, y))

    def predict(self, X):
        predictions = self.sigmoid(X.dot(self.weights) + self.bias)
        print(predictions)
        return np.where(predictions > np.mean(predictions), 1, 0)

# class LogesticRegression2(BaseEstimator, TransformerMixin):
#     def __init__(self, *, param=1):
#         self.param = param
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         return np.full(shape=len(X), fill_value=self.param)
#

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = data.data
    y = data.target

    print(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit + transform training data
    X_test = scaler.transform(X_test)
    clf = LogisticRegressionCustom(logging=True)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print(clf.predict(X_test))
    print(y_test)
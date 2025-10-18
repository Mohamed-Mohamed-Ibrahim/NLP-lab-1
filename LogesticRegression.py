import numpy as np


class LogisticRegressionCustom():

    def __init__(self, lr=0.001, epochs=100, threshold=0.5, logging=False, reg_lambda=0.01, patience=5):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.logging = logging
        self.eps = 1e-5
        self.reg_lambda = reg_lambda
        self.patience = patience

    def sigmoid(self, z):
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),       # +ve values
            np.exp(z) / (1 + np.exp(z)) # -ve values
        )

    def score(self, X, y):
        predictions = self.sigmoid(X.dot(self.weights) + self.bias)
        predictions = np.clip(predictions, self.eps, 1 - self.eps)
        return -np.mean(y * np.log2(predictions) + (1 - y) * np.log2(1 - predictions))

    def fit(self, X, y):
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0
        prev_loss = 0
        early_stopping_counter = 0

        for epoch in range(self.epochs):
            predictions = self.sigmoid(X.dot(self.weights) + self.bias)
            loss = predictions - y
            self.weights -= self.lr * (X.T.dot(loss) / n_samples + self.reg_lambda * self.weights)
            self.bias -= self.lr * loss.mean()

            bce = self.score(X, y)

            if np.abs(bce - prev_loss) < self.eps:
                if early_stopping_counter < self.patience:
                    early_stopping_counter += 1
                else:
                    break
            else:
                early_stopping_counter = 0
            prev_loss = bce

            if self.logging and epoch % 10 == 0:
                print("epoch:", epoch, "loss:", bce)

    def predict(self, X):
        predictions = self.sigmoid(X.dot(self.weights) + self.bias)
        return np.where(predictions > self.threshold, 1, 0)


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegressionCustom(epochs=1000, logging=True)
    ok = LogisticRegression()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit + transform training data
    X_test = scaler.transform(X_test)
    clf.fit(X_train, y_train)
    ok.fit(X_train, y_train)
    print(np.sum(clf.predict(X_test) == y_test))
    print(np.sum(ok.predict(X_test) == y_test))

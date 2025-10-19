import numpy as np
np.random.seed(42)


class LogisticRegressionCustom:

    def __init__(self, lr=0.01, epochs=100, threshold=0.5, logging=False, reg_lambda=0.01, patience=10, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.logging = logging
        self.eps = 1e-5
        self.reg_lambda = reg_lambda
        self.patience = patience
        self.batch_size = batch_size

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
        self.weights = np.zeros(n_features)
        self.bias = 0
        best_loss = float('inf')
        early_stopping_counter = 0
        for epoch in range(self.epochs):

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            for i in range(0, n_samples, self.batch_size):

                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                predictions = self.sigmoid(X_batch.dot(self.weights) + self.bias)
                loss = predictions - y_batch
                self.weights -= self.lr * ((X_batch.T.dot(loss) / len(y_batch)) + self.reg_lambda * self.weights)
                self.bias -= self.lr * loss.mean()

            bce = self.score(X, y)

            if bce < best_loss:
                if early_stopping_counter < self.patience:
                    early_stopping_counter += 1
                else:
                    break
            else:
                early_stopping_counter = 0
                best_loss = bce

            if self.logging and epoch % 10 == 0:
                print("epoch:", epoch+1, "loss:", bce)

    def predict(self, X):
        predictions = self.sigmoid(X.dot(self.weights) + self.bias)
        return np.where(predictions > self.threshold, 1, 0)
    def predict_prob(self, X):
        return self.sigmoid(X.dot(self.weights) + self.bias)


def confusion_matrix_custom(predicts, labels):

    n_classes = len(np.unique(labels))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    precision_vector = np.zeros(n_classes)
    recall_vector = np.zeros(n_classes)
    f1_vector = np.zeros(n_classes)

    for i in range(n_classes):
        for j in range(n_classes):
            cm[i][j] = np.sum((predicts == i) & (labels == j))

    with np.errstate(divide='ignore', invalid='ignore'):  # avoids ZeroDivision warnings
        precision_vector = np.diag(cm) / cm.sum(axis=0)
        recall_vector = np.diag(cm) / cm.sum(axis=1)
        f1_vector = 2 * precision_vector * recall_vector / (precision_vector + recall_vector)

    precision = np.nan_to_num(precision_vector).mean()
    recall = np.nan_to_num(recall_vector).mean()
    f1 = np.nan_to_num(f1_vector).mean()

    return cm, precision, recall, f1

class OneVSAllCustom:
    def __init__(self, base_classifier_class, params: dict={}):
        self.base_classifier_class = base_classifier_class
        self.params = params
        self.clfs = {}

    def fit(self, X, y):
        self.unique_labels = np.unique(y)
        for label in self.unique_labels:
            labels = (y == label).astype(int)

            clf = LogisticRegressionCustom(**self.params)

            clf.fit(X, labels)

            self.clfs[label] = clf


    def predict(self, X):
        scores = []

        for label in self.unique_labels:
            score = self.clfs[label].predict_prob(X)
            scores.append(score)
        scores = np.array(scores)
        return self.unique_labels[np.argmax(scores.T, axis=1)]

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = OneVSAllCustom(LogisticRegressionCustom, {"logging" : True, "patience":20})
    ok = LogisticRegression()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit + transform training data
    X_test = scaler.transform(X_test)
    clf.fit(X_train, y_train)
    ok.fit(X_train, y_train)
    print(np.sum(clf.predict(X_test) == y_test))
    print(np.sum(ok.predict(X_test) == y_test))
    print(confusion_matrix(clf.predict(X_test), y_test))
    print(precision_score(clf.predict(X_test), y_test, average='macro'))
    print(recall_score(clf.predict(X_test), y_test, average='macro'))
    print(f1_score(clf.predict(X_test), y_test, average='macro'))
    cm, precision, recall, f1= confusion_matrix_custom(clf.predict(X_test), y_test)
    print(cm, precision, recall, f1)
    cm, precision, recall, f1= confusion_matrix_custom(ok.predict(X_test), y_test)
    print(cm, precision, recall, f1)

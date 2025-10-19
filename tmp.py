
from nbformat import v4, writes

nb = v4.new_notebook()
nb.cells = [
    v4.new_markdown_cell("# Breast Cancer Classification Notebook"),
    v4.new_code_cell("""from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegressionCustom(epochs=1000, logging=True)
ok = LogisticRegression()
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf.fit(X_train, y_train)
ok.fit(X_train, y_train)

print(np.sum(clf.predict(X_test) == y_test))
print(np.sum(ok.predict(X_test) == y_test))
print(confusion_matrix(ok.predict(X_test), y_test), 
      precision_score(ok.predict(X_test), y_test, average='macro'), 
      recall_score(ok.predict(X_test), y_test), 
      f1_score(ok.predict(X_test), y_test))
print(precision_score(ok.predict(X_test), y_test, average='macro'))
print(recall_score(ok.predict(X_test), y_test, average='macro'))
print(f1_score(ok.predict(X_test), y_test, average='macro'))

cm, precision, recall, f1 = confusion_matrix_custom(ok.predict(X_test), y_test)
print(cm, precision, recall, f1)""")
]

with open('/mnt/data/breast_cancer_notebook.ipynb','w') as f:
    f.write(writes(nb))

print("Notebook created at /mnt/data/breast_cancer_notebook.ipynb")

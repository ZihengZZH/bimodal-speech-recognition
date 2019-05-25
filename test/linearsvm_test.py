import unittest
import numpy as np
from sklearn import datasets
from src.linearsvm import LinearSVM
from src.utility import load_cuave
from sklearn.model_selection import train_test_split


class LinearSVM_test(unittest.TestCase):
    def test_iris_svm(self):
        iris = datasets.load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        iris_svm = LinearSVM('iris')
        iris_svm.train(X_train, y_train)
        iris_svm.test(X_test, y_test)

if __name__ == '__main__':
    unittest.main()
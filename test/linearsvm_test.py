import unittest
from src.linearsvm import LinearSVM


class LinearSVM_test(unittest.TestCase):
    def test_iris_svm(self):
        iris_svm = LinearSVM('cuave', 1, 1, toy_data=True)
        iris_svm.tune()
        iris_svm.train()
        iris_svm.test()


if __name__ == '__main__':
    unittest.main()
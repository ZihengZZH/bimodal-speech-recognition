import unittest
import numpy as np
from src.boltzmann import RBM


class Boltzmann_test(unittest.TestCase):
    def test_RBM(self):
        test_seq = np.array([[0,1,1,0], [0,1,0,0], [0,0,1,1]])
        RBM_instance = RBM(4, 3, 0.1, 100)
        RBM_instance.train(test_seq)



if __name__ == "__main__":
    unittest.main()
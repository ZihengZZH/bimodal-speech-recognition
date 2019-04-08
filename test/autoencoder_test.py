import unittest
from src.autoencoder import Autoencoder


class AutoencoderTest(unittest.TestCase):
    def mnist_test(self):
        mnist_ae = Autoencoder('12','12')
        mnist_ae.build_model()
        mnist_ae.train_model()
        mnist_ae.vis_model()


if __name__ == "__main__":
    unittest.main()

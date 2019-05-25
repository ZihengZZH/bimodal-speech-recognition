import unittest
from src.autoencoder import Autoencoder
from src.utility import flatten_data


class AutoencoderTest(unittest.TestCase):
    def mnist_test(self):
        from keras.datasets import mnist
        print("running autoencoders on MNIST data")
        (X_train, _), (X_test, _) = mnist.load_data()
        
        assert X_train.shape[1:] == X_test.shape[1:]

        X_train = flatten_data(X_train)
        X_test = flatten_data(X_test)

        assert X_train.shape == X_test.shape 

        mnist_ae = Autoencoder('12','12', X_train.shape[1])
        mnist_ae.build_model()
        mnist_ae.train_model(X_train, X_test)
        mnist_ae.vis_model(X_test)


if __name__ == "__main__":
    unittest.main()

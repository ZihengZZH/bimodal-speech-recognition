import unittest
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from src.boltzmann import RBM, BBRBM, GBRBM
from src.utility import visualize_reconstruction


class Boltzmann_test(unittest.TestCase):
    def test_BBRBM(self):
        mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
        mnist_images = mnist.train.images
        bbrbm = BBRBM(use_tqdm=True)
        # errs = bbrbm.fit(mnist_images)
        # bbrbm.save_weights('mnist_bbrbm')
        # plt.plot(errs)
        # plt.show()
        bbrbm.load_weights('mnist_bbrbm')
        image = mnist_images[1]
        image_recon = bbrbm.reconstruct(image.reshape(1,-1))
        visualize_reconstruction(image, image_recon, 28)
    
    def test_GBRBM(self):
        mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
        mnist_images = mnist.train.images
        gbrbm = GBRBM(use_tqdm=True)
        # errs = gbrbm.fit(mnist_images)
        # gbrbm.save_weights('mnist_gbrbm')
        # plt.plot(errs)
        # plt.show()
        gbrbm.load_weights('mnist_gbrbm')
        image = mnist_images[1]
        image_recon = gbrbm.reconstruct(image.reshape(1,-1))
        visualize_reconstruction(image, image_recon, 28)


if __name__ == "__main__":
    unittest.main()
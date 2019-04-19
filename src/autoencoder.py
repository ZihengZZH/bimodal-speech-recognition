import os
import json
import numpy as np
import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from src.utility import load_cuave


class Autoencoder(object):
    """
    The base autoencoder implementation
    ---
    Attributes
    -----------
    _dataset_name: str [private]
        a string indicating the dataset to use
    _modality_name: str [private]
        a string indicating the modality to use
    X_train/X_test: np.array
        a numpy array storing the training/test data (#num x dim)
    y_label: np.array
        a numpy array storing the label (#num x 1)
    encoder/decoder: keras.models.Model()
        the encoder/decoder part of the entire model
    autoencoder: keras.models.Model()
        the entire autoencoder model
    w/h: int
        width/height for input data (if applicable)
    input_dim: int
        dimensionality of input data (w x h)
    hidden_dim: int
        dimensionality of hidden layer (load config)
    batch_size: int
        batch size during training (load config)
    epochs: int
        epochs during training (load config)
    save_dir
        directory for model saving (load config)
    """

    def __init__(self, dataset_name, modality_name):
        """
        # para dataset_name: which dataset to use 
        # para modality_name: which modality to use
        """
        self._dataset_name = dataset_name
        self._modality_name = modality_name
        self.X_train = None
        self.X_test = None
        self.y_label = None
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.w, self.h = 0, 0
        self.config = json.load(open('./config/config.json', 'r'))
        self.hidden_dim = self.config['autoencoder']['hidden_dim']
        self.batch_size = self.config['autoencoder']['batch_size']
        self.epochs = self.config['autoencoder']['epochs']
        self.save_dir = self.config['autoencoder']['save_dir']
        self._prepare_data()
        self.input_dim = self.w * self.h

    def _flatten(self, X, image=True):
        """flatten images to 1D array [private]
        # para X: entire training/test set (#num x dim(75*50))
        # para image: whether image or not
        return X: flattened training/test set
        """
        if image:
            X = X.astype('float32') / 255.
        else:
            X = np.array([x.reshape(self.w*self.h, 1) for x in X])
        X = X.reshape((len(X), np.prod(X.shape[1:])))
        return X

    def _prepare_toy_data(self):
        """prepare some toy data (MNIST) for test of model [private]
        """
        from keras.datasets import mnist
        (X_train, _), (X_test, _) = mnist.load_data()
        self.w, self.h = X_train[0].shape
        self.X_train = self._flatten(X_train)
        self.X_test = self._flatten(X_test)
    
    def _prepare_data(self):
        """prepare data for the model [private]
        X_train/X_test should be modified individually based on modality
        """
        if self._dataset_name == 'cuave':
            mfccs, audio, spec, frames_1, _, labels = load_cuave()
            if self._modality_name == 'mfcc':
                self.X_train, self.X_test, _, _ = train_test_split(mfccs, labels, test_size=0.25)
            elif self._modality_name == 'audio':
                self.X_train, self.X_test, _, _ = train_test_split(audio, labels, test_size=0.25)
            elif self._modality_name == 'spectrogram':
                Sxx = np.array(spec)[:,2]
                self.X_train, self.X_test, _, _ = train_test_split(Sxx, labels, test_size=0.25)
                self.w, self.h = self.X_train[0].shape
                # flatten spectrograms to 1D array
                self.X_train, self.X_test = self._flatten(self.X_train, image=False), self._flatten(self.X_test, image=False)
            elif self._modality_name == 'frame':
                self.X_train, self.X_test, _, _ = train_test_split(frames_1, labels, test_size=0.25)
                self.w, self.h = self.X_train[0].shape
                # flatten frames to 1D array
                self.X_train, self.X_test = self._flatten(self.X_train), self._flatten(self.X_test)
            
        elif self._dataset_name == 'avletter':
            self.X_train = None
            self.y_label = None
        
        print("Training data dimensionality", self.X_train.shape)
        print("Test data dimensionality", self.X_test.shape)
        print("data preparation done")

    def build_model(self):
        """build (deep) autoencoder model
        """
        # an input placeholder
        input_data = Input(shape=(self.input_dim,))

        # encoded representation of the input
        encoded = Dense(self.hidden_dim * 4, activation='relu')(input_data)
        encoded = Dense(self.hidden_dim * 2, activation='relu')(encoded)
        encoded = Dense(self.hidden_dim, activation='relu')(encoded)
        # decoded representation of the input
        decoded = Dense(self.hidden_dim * 2, activation='relu')(encoded)
        decoded = Dense(self.hidden_dim * 4, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)

        # maps an input to its reconstruction
        self.autoencoder = Model(input_data, decoded)
        # maps an input to its encoded representation
        self.encoder = Model(input_data, encoded)

        # an encoded input placeholder
        encoded_input = Input(shape=(self.hidden_dim,))
        # retrieve layer of the autoencoder model
        decoder_layer1 = self.autoencoder.layers[-3]
        decoder_layer2 = self.autoencoder.layers[-2]
        decoder_layer3 = self.autoencoder.layers[-1]
        # maps the encoded representation to the input
        self.decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
        
        # configure the model
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        print(self.autoencoder.summary())
    
    def train_model(self):
        """train (deep) autoencoder model and save to external file
        """
        self.autoencoder.fit(self.X_train, self.X_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True,
                            validation_data=(self.X_test, self.X_test))
        self._save_model()

    def _save_model(self):
        """save (deep) autoencoder model to (indicated) external file [private]
        """
        save_name = os.path.join(self.save_dir, '%s-%s-%d-%s.h5' % (self._dataset_name, self._modality_name, self.hidden_dim, datetime.datetime.now().strftime('%d%m%Y-%H%M%S')))
        self.autoencoder.save_weights(save_name)
    
    def load_model(self):
        """load (deep) autoencoder model from external files
        """
        weights_list = [f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))]
        print(weights_list)
        print("Here are the weights of pre-trained models")
        for idx, name in enumerate(weights_list):
            print("no.%d model with name %s" % (idx, name))
        choose_model = None
        while True:
            try:
                choose_model = input("Please make your choice\t")
                weights_name = weights_list[int(choose_model)]
                self.autoencoder.load_weights(os.path.join(self.save_dir, weights_name))
                break
            except:
                print("Wrong input! Please start over")

    def vis_model(self):
        """visualize original/reconstructed data along with encoded representation
        """
        encoded_repres = self.encoder.predict(self.X_test)      # inference
        decoded_repres = self.decoder.predict(encoded_repres)   # inference
        n = 10 # number to visualize
        plt.figure(figsize=(30, 8))
        
        for i in range(n):
            ax = plt.subplot(3, n, i+1)
            plt.imshow(self.X_test[i].reshape(self.w, self.h))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, i+1+n)
            plt.imshow(encoded_repres[i].reshape(5, 4))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, i+1+n*2)
            plt.imshow(decoded_repres[i].reshape(self.w, self.h))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()


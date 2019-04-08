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
    def __init__(self, dataset_name, modality_name):
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
        self.hidden_dim = self.config['model']['hidden_dim']
        self.batch_size = self.config['model']['batch_size']
        self.epochs = self.config['model']['epochs']
        self.save_dir = self.config['model']['save_dir']
        self._prepare_data()
        self.input_dim = self.w * self.h

    # flatten images to 1D array
    def _flatten(self, X, image=True):
        # para image: whether image or not
        if image:
            X = X.astype('float32') / 255.
        X = X.reshape((len(X), np.prod(X.shape[1:])))
        return X

    def _prepare_toy_data(self):
        from keras.datasets import mnist
        (X_train, _), (X_test, _) = mnist.load_data()
        self.w, self.h = X_train[0].shape
        self.X_train = self._flatten(X_train)
        self.X_test = self._flatten(X_test)
    
    def _prepare_data(self):
        if self._dataset_name == 'cuave':
            mfccs, audio, spec, frames_1, _, labels = load_cuave()
            if self._modality_name == 'mfcc':
                self.X_train, self.X_test, _, _ = train_test_split(mfccs, labels, test_size=0.25)
            elif self._modality_name == 'audio':
                self.X_train, self.X_test, _, _ = train_test_split(audio, labels, test_size=0.25)
            elif self._modality_name == 'spectrogram':
                self.X_train, self.X_test, _, _ = train_test_split(spec, labels, test_size=0.25)
            elif self._modality_name == 'frame':
                self.X_train, self.X_test, _, _ = train_test_split(frames_1, labels, test_size=0.25)
                self.w, self.h = self.X_train[0].shape
                # flatten frames to 1D array
                self.X_train, self.X_test = self._flatten(self.X_train), self._flatten(self.X_test)
            
        elif self._dataset_name == 'avletter':
            self.X_train = None
            self.y_label = None

    def build_model(self):
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
    
    def train_model(self):
        self.autoencoder.fit(self.X_train, self.X_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True,
                            validation_data=(self.X_test, self.X_test))
        self._save_model()

    def _save_model(self):
        save_name = os.path.join(self.save_dir, '%s-e%s.h5' % (datetime.datetime.now().strftime('%d%m%Y-%H%M%S'), str(self.epochs)))
        self.autoencoder.save_weights(save_name)
    
    def load_model(self):
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
        encoded_repres = self.encoder.predict(self.X_test)
        decoded_repres = self.decoder.predict(encoded_repres)
        n = 10
        plt.figure(figsize=(30, 8))
        for i in range(n):
            ax = plt.subplot(3, n, i+1)
            plt.imshow(self.X_test[i].reshape(self.w, self.h))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, i+1+n)
            plt.imshow(encoded_repres[i].reshape(25, 10))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, i+1+n*2)
            plt.imshow(decoded_repres[i].reshape(self.w, self.h))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()


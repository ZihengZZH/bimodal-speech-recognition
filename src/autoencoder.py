import os
import json
import numpy as np
import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.utils import plot_model


class Autoencoder(object):
    """
    Deep Autoencoder
    ---
    Attributes
    -----------

    """
    def __init__(self, dataset_name, modality_name, input_dim):
        # para dataset_name:
        # para modality_name: 
        # para input_dim:
        self.dataset_name = dataset_name
        self.modality_name = modality_name
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.config = json.load(open('./config/config.json', 'r'))
        self.hidden_ratio = self.config['autoencoder']['hidden_ratio']
        self.batch_size = self.config['autoencoder']['batch_size']
        self.epochs = self.config['autoencoder']['epochs']
        self.save_dir = self.config['autoencoder']['save_dir']
        self.input_dim = input_dim
        self.hidden_dim = [
            int(self.input_dim * self.hidden_ratio),
            int(self.input_dim * self.hidden_ratio ** 2),
            int(self.input_dim * self.hidden_ratio ** 3)
        ]

    def build_model(self):
        """build (deep) autoencoder model
        """
        # an input placeholder
        input_data = Input(shape=(self.input_dim, ))

        # encoded representation of the input
        encoded = Dense(self.hidden_dim[0], activation='relu')(input_data)
        encoded = Dense(self.hidden_dim[1], activation='relu')(encoded)
        encoded = Dense(self.hidden_dim[2], activation='relu')(encoded)
        # decoded representation of the input
        decoded = Dense(self.hidden_dim[1], activation='relu')(encoded)
        decoded = Dense(self.hidden_dim[0], activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)

        # maps an input to its reconstruction
        self.autoencoder = Model(inputs=input_data, outputs=decoded)
        # maps an input to its encoded representation
        self.encoder = Model(inputs=input_data, outputs=encoded)

        # an encoded input placeholder
        encoded_input = Input(shape=(self.hidden_dim[2], ))
        # retrieve layer of the autoencoder model
        decoder_layer1 = self.autoencoder.layers[-3]
        decoder_layer2 = self.autoencoder.layers[-2]
        decoder_layer3 = self.autoencoder.layers[-1]
        # maps the encoded representation to the input
        self.decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
        
        # configure the model
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        print(self.autoencoder.summary())
        plot_model(self.autoencoder, show_shapes=True, to_file='./images/autoencoder_%s_%s.png' % (self.dataset_name, self.modality_name))
    
    def train_model(self, X_train):
        """train (deep) autoencoder model and save to external file
        """
        self.autoencoder.fit(X_train, X_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True)
        self.save_model()

    def save_model(self):
        """save (deep) autoencoder model to (indicated) external file
        """
        save_name = os.path.join(self.save_dir, '%s-%s-%s.h5' % (self.dataset_name, self.modality_name, datetime.datetime.now().strftime('%d%m-%H%M')))
        self.autoencoder.save_weights(save_name)
    
    def load_model(self):
        """load (deep) autoencoder model from external files
        """
        weights_list = [f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))]
        print("--" * 20)
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

    def vis_model(self, X_test):
        """visualize original/reconstructed data along with encoded representation
        """
        encoded_repres = self.encoder.predict(X_test)           # inference
        decoded_repres = self.decoder.predict(encoded_repres)   # inference
        w, h = decoded_repres.shape[-2:]
        n = 10 # number to visualize
        plt.figure(figsize=(30, 8))
        
        for i in range(n):
            ax = plt.subplot(3, n, i+1)
            plt.imshow(X_test[i].reshape(w, h))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, i+1+n)
            plt.imshow(encoded_repres[i].reshape(5, 4))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, i+1+n*2)
            plt.imshow(decoded_repres[i].reshape(w, h))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

    def transform(self, X_input):
        """transform inputs to latent representations
        """
        X_encoded = self.encoder.predict(X_input)
        np.save(os.path.join(self.config['autoencoder']['encoded'], 'encoded_repres_%s_%s' % (self.dataset_name, self.modality_name)), X_encoded)
        return X_encoded
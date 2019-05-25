import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate
from keras.utils import plot_model
from src.autoencoder import Autoencoder


class AutoencoderBimodal(Autoencoder):
    """
    Bimodal Deep Autoencoder
    ---
    Attribute
    ----------

    """
    def __init__(self, dataset_name, arch_name, input_dim_A, input_dim_V):
        # para dataset_name:
        # para modality_name:
        # para input_dim_A:
        # para input_dim_V:
        Autoencoder.__init__(self, dataset_name, 'bimodal_%s' % arch_name, 0)
        self.save_dir = self.config['autoencoder']['save_dir_bimodal']
        self.input_dim_A = input_dim_A
        self.input_dim_V = input_dim_V
        self.hidden_dim_A = [
            int(self.input_dim_A * self.hidden_ratio),
            int(self.input_dim_A * self.hidden_ratio ** 2),
        ]
        self.hidden_dim_V = [
            int(self.input_dim_V * self.hidden_ratio),
            int(self.input_dim_V * self.hidden_ratio ** 2),
        ]
        self.hidden_dim_shared = int(self.hidden_dim_A[1] / 4 + self.hidden_dim_V[1] / 4)

    def build_model(self):
        """build bimodal (deep) autoencoder model
        """
        input_data_A = Input(shape=(self.input_dim_A, ), name='input_A')
        input_data_V = Input(shape=(self.input_dim_V, ), name='input_V')
        # encoded_input = Input(shape=(self.hidden_dim_shared, ), name='shared_repres')

        encoded_A = Dense(self.hidden_dim_A[0], activation='relu', name='encoded_A_1')(input_data_A)
        encoded_V = Dense(self.hidden_dim_V[0], activation='relu', name='encoded_V_1')(input_data_V)

        encoded_A = Dense(self.hidden_dim_A[1], activation='relu', name='encoded_A_2')(encoded_A)
        encoded_V = Dense(self.hidden_dim_V[1], activation='relu', name='encoded_V_2')(encoded_V)

        shared = Concatenate(axis=1, name='concat')([encoded_A, encoded_V])
        encoded = Dense(self.hidden_dim_shared, activation='relu', name='shared_layer')(shared)

        decoded_A = Dense(self.hidden_dim_A[1], activation='relu', name='decoded_A_2')(encoded)
        decoded_V = Dense(self.hidden_dim_V[1], activation='relu', name='decoded_V_2')(encoded)

        decoded_A = Dense(self.hidden_dim_A[0], activation='relu', name='decoded_A_1')(decoded_A)
        decoded_V = Dense(self.hidden_dim_V[0], activation='relu', name='decoded_V_1')(decoded_V)

        decoded_A = Dense(self.input_dim_A, activation='sigmoid', name='decoded_A')(decoded_A)
        decoded_V = Dense(self.input_dim_V, activation='sigmoid', name='decoded_V')(decoded_V)

        self.autoencoder = Model(inputs=[input_data_A, input_data_V], outputs=[decoded_A, decoded_V])
        self.encoder = Model(inputs=[input_data_A, input_data_V], outputs=encoded)

        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        print(self.autoencoder.summary())
        plot_model(self.autoencoder, show_shapes=True, to_file='./images/autoencoder_%s_%s.png' % (self.dataset_name, self.modality_name))

    def train_model(self, X_train_A, X_train_V):
        """train bimodal (deep) autoencoder model and save to external file
        """
        self.autoencoder.fit([X_train_A, X_train_V],
                            [X_train_A, X_train_V],
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            shuffle=True)
        self.save_model()

    def transform(self, X_input_A, X_input_V):
        """transform bimodal inputs to latent representations
        """
        X_encoded = self.encoder.predict([X_input_A, X_input_V])
        np.save(os.path.join(self.config['autoencoder']['encoded'], 'encoded_repres_%s_%s' % (self.dataset_name, self.modality_name)), X_encoded)
        return X_encoded
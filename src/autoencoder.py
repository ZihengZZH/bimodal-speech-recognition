import json
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from keras.utils import plot_model
from src.utility import load_cuave

config = json.load(open('./config/config.json'))
hidden = config['model']['hidden']
epochs = config['model']['epochs']
plot = config['model']['plot']


class Autoencoder():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def __prepare_data(self):
        mfccs, frames_1, frames_2, labels = load_cuave()

    def build_model(self):
        pass
    
    def train_model(self):
        pass

    def __save_model(self):
        pass
    
    def __load_model(self):
        pass

    def test_model(self):
        pass
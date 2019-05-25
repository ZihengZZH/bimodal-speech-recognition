import numpy as np
import matplotlib.pyplot as plt
from src.utility import load_data, flatten_data
from src.autoencoder import Autoencoder
from src.autoencoder_bimodal import AutoencoderBimodal
from src.linearsvm import LinearSVM
from src.forest import tree_models
from sklearn.model_selection import train_test_split



def baseline(dataset):
    audio, _, _, labels = load_data(dataset, 'audio', 'frame', verbose=True)
    
    X = flatten_data(audio, image=False)
    y = labels[:,0]

    print("--" * 20)
    print("processed data shape", X.shape)
    print("processed label shape", y.shape)
    print("--" * 20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    assert X_train.shape[1] == X_test.shape[1]

    test_AE = Autoencoder(dataset, 'audio', X_train.shape[1])
    test_AE.build_model()
    test_AE.train_model(X_train)
    
    X_encoded_train = test_AE.transform(X_train)
    X_encoded_test = test_AE.transform(X_test)

    test_SVM = LinearSVM('%s_baseline_%s' % (dataset, 'audio'))
    test_SVM.train(X_encoded_train, y_train)
    test_SVM.test(X_encoded_test, y_test)


def bimodal_fusion(dataset):
    pass


def cross_modality(dataset):
    mfccs, frames_1, frames_2, labels = load_data(dataset, 'mfcc', 'frame', verbose=True)

    X_A = np.vstack((flatten_data(mfccs, image=False), flatten_data(mfccs, image=False)))
    X_V = np.vstack((flatten_data(frames_1, image=True), flatten_data(frames_2, image=True)))
    y = np.hstack((labels[:,0], labels[:,0]))

    print("--" * 20)
    print("processed data A shape", X_A.shape)
    print("processed data V shape", X_V.shape)
    print("processed label shape", y.shape)
    print("--" * 20)

    assert X_A.shape[0] == X_V.shape[0] == y.shape[0]

    no_sample = y.shape[0]
    pivot = int(no_sample * 0.67)

    X_train_A, X_test_A = X_A[:pivot, :], X_A[pivot:, :]
    X_train_V, X_test_V = X_V[:pivot, :], X_V[pivot:, :]
    y_train, y_test = y[:pivot], y[pivot:]

    assert X_train_A.shape[1] == X_test_A.shape[1]
    assert X_train_V.shape[1] == X_test_V.shape[1]
    assert X_train_A.shape[0] == X_train_V.shape[0] == len(y_train)
    assert X_test_A.shape[0] == X_test_V.shape[0] == len(y_test)

    test_AE = AutoencoderBimodal(dataset, 'cross_mfcc', X_train_A.shape[1], X_train_V.shape[1])
    test_AE.build_model()
    test_AE.train_model(X_train_A, X_train_V)
    
    X_encoded_train = test_AE.transform(X_train_A, X_train_V)
    X_encoded_test = test_AE.transform(X_test_A, X_test_V)

    test_SVM = LinearSVM('%s_baseline_%s' % (dataset, 'cross_mfcc'))
    test_SVM.train(X_encoded_train, y_train)
    test_SVM.test(X_encoded_test, y_test)


def shared_repres():
    pass


def main():
    cross_modality('cuave')

if __name__ == "__main__":
    main()
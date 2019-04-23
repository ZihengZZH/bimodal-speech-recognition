import unittest
import numpy as np
from src.linearsvm import LinearSVM
from src.utility import load_cuave
from sklearn.model_selection import train_test_split


class LinearSVM_test(unittest.TestCase):
    def test_iris_svm(self):
        # iris_svm = LinearSVM('cuave', 1, 1, toy_data=True)
        # iris_svm.tune()
        # iris_svm.train()
        # iris_svm.test()
        pass
    
    def test_cuave_mfcc(self):
        print("testing CUAVE mfcccs alone")
        mfccs, _, _, _, labels = load_cuave()
        mfccs = np.reshape(mfccs, (mfccs.shape[0], int(mfccs.shape[1]*mfccs.shape[2])))
        labels = labels[:,0]
        print(mfccs.shape, labels.shape)
        X_train, X_test, y_train, y_test = train_test_split(mfccs, labels, test_size=0.25, random_state=1)
        svm = LinearSVM('cuave', mfccs, labels)
        svm.tune()
        svm.train()
        svm.test()

    def test_cuave_audio(self):
        print("testing CUAVE audio alone")
        _, audio, _, _, labels = load_cuave()
        audio = np.reshape(audio, (audio.shape[0], int(audio.shape[1]*audio.shape[2])))
        labels = labels[:,0]
        print(audio.shape, labels.shape)
        X_train, X_test, y_train, y_test = train_test_split(audio, labels, test_size=0.25, random_state=1)
        svm = LinearSVM('cuave', audio, labels)
        svm.tune()
        svm.train()
        svm.test()
    
    def test_cuave_video(self):
        print("testing CUAVE frames alone")
        _, _, frames_1, _, labels = load_cuave()
        frames_1 = np.reshape(frames_1, (frames_1.shape[0], int(frames_1.shape[1]*frames_1.shape[2]*frames_1.shape[3])))
        labels = labels[:,0]
        print(frames_1.shape, labels.shape)
        X_train, X_test, y_train, y_test = train_test_split(frames_1, labels, test_size=0.25, random_state=1)
        svm = LinearSVM('cuave', frames_1, labels)
        svm.tune()
        svm.train()
        svm.test()


if __name__ == '__main__':
    unittest.main()
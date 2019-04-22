import os
import json
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
# ignore Future Warning
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class LinearSVM:
    def __init__(self, dataset, X, y, toy_data=False):
        if dataset != 'cuave' and dataset != 'avletter':
            raise ValueError("wrong \'dataset\' argument")

        self.dataset = dataset
        self.toy_data = toy_data
        self.config = json.load(open('./config/config.json', 'r'))
        self.C = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self._prepare_data(X, y)
    
    def _prepare_data(self, X, y):
        if self.toy_data:
            from sklearn import datasets
            iris = datasets.load_iris()
            X = iris.data[:,:2]
            y = iris.target
            indices = np.random.permutation(len(X))
            test_size = 15
            self.X_train = X[indices[:-test_size]]
            self.y_train = y[indices[:-test_size]]
            self.X_test = X[indices[-test_size:]]
            self.y_test = y[indices[-test_size:]]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)


    def train(self):
        if not self.C:
            print("-------------------WARNING-------------------")
            print("please tune the hyperparameters for SVM first")
            return 
        
        self.model = svm.SVC(kernel='linear', C=self.C)
        self.model.fit(self.X_train, self.y_train)

    def test(self):
        if not self.model:
            print("-------------------WARNING-------------------")
            print("please train the SVM model first")
            return 

        y_pred = self.model.predict(self.X_test)
        print(metrics.classification_report(self.y_test, y_pred))
        print("\noverall accuracy: %.4f" % metrics.accuracy_score(self.y_test, y_pred))

    def tune(self):
        if self.C:
            print("-------------------WARNING-------------------")
            print("please proceed as the penalty has been tuned")
            return 
        
        C_range = 10. ** np.arange(-5, 5, step=1)
        grid = GridSearchCV(svm.SVC(), [{'C': C_range, 'kernel': ['linear']}], cv=10, n_jobs=-1, verbose=1)
        grid.fit(self.X_train, self.y_train)

        print("\nbest parameters set found on development set\n")
        print(grid.best_params_)

        self.C = grid.best_params_['C']
